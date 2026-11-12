/* Host-only ABI adapter. The actual control implementation is compiled directly
 * from the electrical repository; no alternate PD/compensation/limit formula.
 * Starting READY models an already available takeover opportunity, not an LQR
 * simulation or proof of a real handover. Receipt ages are explicit hypotheses.
 */
#include "rl_control.h"
#include "rl_capture.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "leg_const.hpp"
void getPhi(float,float,float,float,float,float,float,float *,float *,float *,float *);

int motion_fw_geometry(const float *q4,float *length2,float *spread2)
{
    const float target[6]={q4[0],q4[1],0,q4[2],q4[3],0};
    float joints[4],wheels[2],p2,p3,theta;
    if(RLMotion31MapPolicyTargets(target,joints,wheels)!=0)return -1;
    getPhi(joints[0],joints[1],L_1,L_2,L_3,L_4,L_5,&p2,&p3,&length2[0],&theta);
    getPhi(joints[2],joints[3],L_1,L_2,L_3,L_4,L_5,&p2,&p3,&length2[1],&theta);
    spread2[0]=joints[0]-joints[1];spread2[1]=joints[2]-joints[3];
    return isfinite(length2[0])&&isfinite(length2[1])&&length2[0]>0&&length2[1]>0?0:-1;
}

typedef struct {
    RLControlState state;
    RLControlLegRamp ramp;
    RLShadowSnapshot actor;
    RLShadowCommand command;
    RLControlFrame frame;
    uint32_t policy_sequence;
} Bridge;

unsigned motion_fw_abi_version(void){return 1u;}
unsigned motion_fw_capture_record_bytes(void){return sizeof(RLCaptureRecord);}
unsigned motion_fw_capture_buffer_bytes(void){return sizeof(RLCaptureBuffer);}

void *motion_fw_create(void)
{
    Bridge *b=calloc(1,sizeof(*b));
    if(b){b->ramp.completed=1u;b->state.timing=1u;b->state.ready=1u;
          b->state.epoch=1u;b->state.wait_start_tick=(uint32_t)-5000;}
    return b;
}
void motion_fw_destroy(void *context){free(context);}

/* Input: native gyro3, gravity3, LEG q4, CH dq6. No raw motor coordinates. */
static void fill(RLShadowSnapshot *s,uint32_t tick,uint32_t sensor_tick,
                 const float *native16,const float *command3,
                 const float *length2,const float *spread2)
{
    memset(s,0,sizeof(*s));s->tick_ms=tick;s->sequence=tick+1u;s->chassis_ready=1u;
    s->rl_active=1u;s->feedback_valid_mask=0x3fu;s->imu_update_sequence=1u;
    s->imu_update_tick_ms=sensor_tick;
    for(unsigned i=0;i<6u;++i)s->feedback_receive_tick_ms[i]=sensor_tick;
    memcpy(s->motion.gyro_model_rad_s,native16,3*sizeof(float));
    memcpy(s->motion.gravity_model,native16+3,3*sizeof(float));
    memcpy(s->motion.joint_position_rad,native16+6,4*sizeof(float));
    memcpy(s->motion.joint_velocity_rad_s,native16+10,6*sizeof(float));
    s->motion.command_vx_m_s=command3[0];s->motion.command_yaw_rad_s=command3[1];
    s->motion.command_height_m=command3[2];
    memcpy(s->leg_length_m,length2,2*sizeof(float));memcpy(s->leg_spread_rad,spread2,2*sizeof(float));
}

int motion_fw_observe(void *context,uint32_t tick,uint32_t sensor_tick,
                      const float *native16,const float *command3,float *out31)
{
    Bridge *b=context;if(!b)return -1;
    const float length[2]={.28f,.28f},spread[2]={0};
    fill(&b->actor,tick,sensor_tick,native16,command3,length,spread);
    b->actor.sequence=++b->policy_sequence;
    RLControlFillPrevious(&b->state,&b->actor);
    if(!RLControlFeedbackFresh(&b->actor,tick))return -2;
    return RLMotion31BuildObservation(&b->actor.motion,out31);
}

int motion_fw_publish(void *context,uint32_t now,const float *action6)
{
    Bridge *b=context;if(!b)return -1;
    RLShadowCommand c={0};
    c.snapshot_sequence=b->actor.sequence;c.snapshot_tick_ms=b->actor.tick_ms;c.output_tick_ms=now;
    c.control_epoch=b->actor.control_epoch;c.previous_source=b->actor.previous_source;
    c.previous_sequence=b->actor.adopted_sequence;
    memcpy(c.action,action6,sizeof(c.action));
    if(RLMotion31BuildPreview(&b->actor.motion,c.action,&c.preview)!=0)return -2;
    c.valid=1;b->command=c;return 0;
}

/* output[0:6] motor commands mapped to model coordinates, [6:12] spring
 * compensation in model coordinates (not actual spring force), [12:18] PD raw,
 * [18:24] PD limited, [24:30] adopted target. Metadata is separate uint32.
 */
int motion_fw_pd(void *context,uint32_t tick,uint32_t sensor_tick,
                 const float *native16,const float *command3,
                 const float *length2,const float *spread2,float *output30,uint32_t *metadata8)
{
    Bridge *b=context;if(!b)return -1;
    RLShadowSnapshot s;
    fill(&s,tick,sensor_tick,native16,command3,length2,spread2);
    RLControlFillPrevious(&b->state,&s);
    const int rc=RLControlExecute(&b->state,&b->ramp,tick,1,&s,&b->command,&b->frame);
    memset(output30,0,30*sizeof(float));
    if(!rc){
        const float *m=b->frame.final_motor_nm,*c=b->frame.spring_motor_nm;
        output30[0]=-m[3];output30[1]=m[2];output30[3]=-m[1];output30[4]=-m[0];
        output30[2]=-b->frame.wheel_current_raw[1]/RL_MOTION31_WHEEL_CURRENT_PER_NM;
        output30[5]= b->frame.wheel_current_raw[0]/RL_MOTION31_WHEEL_CURRENT_PER_NM;
        output30[6]=-c[3];output30[7]=c[2];output30[9]=-c[1];output30[10]=-c[0];
        memcpy(output30+12,b->frame.tau_pd_nm,6*sizeof(float));
        memcpy(output30+18,b->frame.tau_limited_nm,6*sizeof(float));
        memcpy(output30+24,b->frame.adopted_target,6*sizeof(float));
    }
    metadata8[0]=b->frame.owner;metadata8[1]=b->frame.fault;metadata8[2]=b->state.epoch;
    metadata8[3]=b->state.adopted_sequence;metadata8[4]=b->state.adopted_tick_ms;
    metadata8[5]=s.previous_source;metadata8[6]=s.adopted_sequence;metadata8[7]=tick;
    return rc;
}
