/* Host-only causal packet/driver adapter. Production core and drivers are linked
 * unchanged. HAL/RTOS, CAN delivery and internal motor measurements are explicit
 * simulation boundaries, not an emulation of the STM32 or a flashed build. */
#include "firmware_bridge.c"
#include "DMmotor.h"
#include "dji_motor.h"
#include "task.h"
#include "general_def.h"
#include <stdarg.h>
#include "sampling_source_constants.inc"

extern void DMMotorDecode(CANInstance *);
extern float uint_to_float(int,float,float,int);
FDCAN_HandleTypeDef hfdcan1={1},hfdcan2={2},hfdcan3={3};
static DMMotorInstance *dm[4]; /* lf,lb,rf,rb */
static DJIMotorInstance *dji[2]; /* geometric left,right */
static float wheel_current[2],wire_model[6];
static uint32_t clock_ms,wire_tick[6],wire_sequence[6];
static unsigned critical_depth,task_calls,initialized;
TickType_t xTaskGetTickCount(void){return clock_ms;}
TickType_t xTaskGetTickCountFromISR(void){return clock_ms;}
void host_driver_enter_critical(void){critical_depth++;}
void host_driver_exit_critical(void){if(critical_depth)critical_depth--;}
void host_driver_log(const char *fmt,...){(void)fmt;}
float DWT_GetDeltaT(uint32_t *last){float dt=(uint32_t)(clock_ms-*last)*.001f;*last=clock_ms;return dt;}
DaemonInstance *DaemonRegister(Daemon_Init_Config_s *c)
{DaemonInstance *d=calloc(1,sizeof(*d));if(d)d->temp_count=c->reload_count;return d;}
void DaemonReload(DaemonInstance *d){if(d)d->temp_count=2;}
uint8_t DaemonIsOnline(DaemonInstance *d){return d&&d->temp_count;}
CANInstance *CANRegister(CAN_Init_Config_s *cfg)
{CANInstance *c=calloc(1,sizeof(*c));if(c){c->can_handle=cfg->can_handle;c->tx_id=cfg->tx_id;
 c->rx_id=cfg->rx_id;c->id=cfg->id;c->can_module_callback=cfg->can_module_callback;}return c;}
void PIDInit(PIDInstance *p,const PID_Init_Config_s *c){p->Kp=c->Kp;p->Ki=c->Ki;p->Kd=c->Kd;}
float PIDCalculate(PIDInstance *p,float measured,float target)
{return p->Kp*(target-measured);} /* production wheel PID gains are all zero */

uint8_t CANTransmit(CANInstance *c,float timeout)
{
    (void)timeout;
    const unsigned channel[4]={4,3,1,0};const float sign[4]={-1,-1,1,-1};
    for(unsigned i=0;i<4;i++)if(dm[i]!=NULL && c==dm[i]->motor_can_instance){
        unsigned ch=channel[i];int word=((c->tx_buff[6]&15)<<8)|c->tx_buff[7];
        wire_model[ch]=sign[i]*uint_to_float(word,-40,40,12);
        wire_tick[ch]=clock_ms;wire_sequence[ch]++;return 1;
    }
    if(c->txconf.Identifier==0x200){
        unsigned side=c->can_handle==&hfdcan2?0u:1u;
        unsigned byte=side==0?0u:2u,ch=side==0?5u:2u;
        int16_t raw=(int16_t)((c->tx_buff[byte]<<8)|c->tx_buff[byte+1]);
        wire_model[ch]=(side==0?1.f:-1.f)*raw/RL_MOTION31_WHEEL_CURRENT_PER_NM;
        wire_tick[ch]=clock_ms;wire_sequence[ch]++;
    }
    return 1; /* captures enqueue/packing only, never claims remote acceptance */
}

/* Exact production function body, extracted by the builder, including its
 * static counter. Nothing from the electrical working tree is rewritten. */
#include "sampling_motor_task.inc"

int motion_io_reset(void)
{
    if(!initialized){
        for(unsigned i=0;i<4;i++){
            Motor_Init_Config_s c={0};c.can_init_config.can_handle=i<2?&hfdcan2:&hfdcan1;
            c.can_init_config.tx_id=i%2+1;c.can_init_config.rx_id=0x11+i%2;
            c.motor_set=(MotorSet){6.283f,45,500,5,40,1};dm[i]=DMMotorInit(&c);
            if(!dm[i])return -1;
        }
        for(unsigned i=0;i<2;i++){
            Motor_Init_Config_s c={0};c.can_init_config.can_handle=i==0?&hfdcan2:&hfdcan1;
            c.can_init_config.tx_id=i==0?1:2;c.motor_type=M3508;
            c.controller_setting_init_config.feedforward_flag=CURRENT_FEEDFORWARD;
            c.controller_setting_init_config.outer_loop_type=SPEED_LOOP;
            c.controller_setting_init_config.close_loop_type=SPEED_LOOP;
            c.controller_param_init_config.current_feedforward_ptr=wheel_current+i;
            dji[i]=DJIMotorInit(&c);if(!dji[i])return -1;
        }
        initialized=1;
    }
    /* Restore the exact production counter phase without patching its body.
     * An alignment-only call is outside physical time and its output discarded. */
    if(task_calls%2){MotorControlTask();task_calls++;}
    for(unsigned i=0;i<4;i++){
        memset(&dm[i]->measure,0,sizeof(dm[i]->measure));memset(&dm[i]->ctrl,0,sizeof(dm[i]->ctrl));
        dm[i]->measure.state=1;dm[i]->online=1;dm[i]->stop_flag=MOTOR_ENALBED;
        dm[i]->feedback_sequence=0;dm[i]->feedback_receive_tick_ms=0;
    }
    for(unsigned i=0;i<2;i++){
        memset(&dji[i]->measure,0,sizeof(dji[i]->measure));dji[i]->stop_flag=MOTOR_ENALBED;
        dji[i]->feedback_sequence=0;dji[i]->feedback_receive_tick_ms=0;wheel_current[i]=0;
    }
    memset(wire_model,0,sizeof(wire_model));memset(wire_tick,0,sizeof(wire_tick));
    memset(wire_sequence,0,sizeof(wire_sequence));return 0;
}

/* Invert the existing calibrated mapping to generate hypothetical encoder
 * packets; these offsets set the quantization grid, not a new calibration. */
int motion_io_motor_coordinates(const float *q4,const float *dq6,float *motor6)
{
    float target[6]={q4[0],q4[1],0,q4[2],q4[3],0},joint[4],wheel[2];
    if(RLMotion31MapPolicyTargets(target,joint,wheel))return -1;
    motor6[0]=LF_ENCODE_OFFSET-joint[0];motor6[1]=LB_ENCODE_OFFSET-joint[1];
    motor6[2]=joint[2]-RF_ENCODE_OFFSET;motor6[3]=joint[3]-RB_ENCODE_OFFSET;
    motor6[4]=dq6[5]*REDUCE_RATIO/DEGREE_2_RAD/RPM_2_ANGLE_PER_SEC;
    motor6[5]=-dq6[2]*REDUCE_RATIO/DEGREE_2_RAD/RPM_2_ANGLE_PER_SEC;
    return 0;
}

int motion_io_leg_receive(unsigned index,unsigned p,unsigned v,uint32_t receive_tick)
{
    if(index>=4||p>65535||v>4095)return -1;
    clock_ms=receive_tick;CANInstance *c=dm[index]->motor_can_instance;c->rx_len=8;
    memset(c->rx_buff,0,8);c->rx_buff[0]=0x10|(c->tx_id&15);
    c->rx_buff[1]=p>>8;c->rx_buff[2]=p;c->rx_buff[3]=v>>4;c->rx_buff[4]=(v&15)<<4;
    DMMotorDecode(c);return 0;
}
int motion_io_wheel_receive(unsigned side,int rpm,uint32_t receive_tick)
{
    if(side>=2||rpm< -32768||rpm>32767)return -1;
    clock_ms=receive_tick;CANInstance *c=dji[side]->motor_can_instance;c->rx_len=8;
    memset(c->rx_buff,0,8);c->rx_buff[2]=(uint16_t)rpm>>8;c->rx_buff[3]=rpm;
    c->can_module_callback(c);return 0;
}
int motion_io_read(float *native10,uint32_t *receipt6,uint32_t *sequence6)
{
    RLMotion31Snapshot s={0};
    float lf=-dm[0]->measure.pos+LF_ENCODE_OFFSET,lb=-dm[1]->measure.pos+LB_ENCODE_OFFSET;
    float rf=dm[2]->measure.pos+RF_ENCODE_OFFSET,rb=dm[3]->measure.pos+RB_ENCODE_OFFSET;
    RLMotion31MapJointAngles(&s,rb,rf,lb,lf,dm[3]->measure.vel,dm[2]->measure.vel,
                            -dm[1]->measure.vel,-dm[0]->measure.vel);
    RLMotion31MapWheelMotorSpeeds(&s,dji[0]->measure.speed_aps/REDUCE_RATIO*DEGREE_2_RAD,
                                   dji[1]->measure.speed_aps/REDUCE_RATIO*DEGREE_2_RAD);
    memcpy(native10,s.joint_position_rad,4*sizeof(float));
    memcpy(native10+4,s.joint_velocity_rad_s,6*sizeof(float));
    const unsigned index[6]={3,2,0,1,0,0};
    for(unsigned ch=0;ch<6;ch++){
        if(ch==2||ch==5){unsigned i=ch==2?1:0;
            receipt6[ch]=dji[i]->feedback_receive_tick_ms;sequence6[ch]=dji[i]->feedback_sequence;
        }else{unsigned i=index[ch];receipt6[ch]=dm[i]->feedback_receive_tick_ms;sequence6[ch]=dm[i]->feedback_sequence;}
    }
    return critical_depth? -1:0;
}
int motion_io_send(uint32_t tick,const float *demand6,float *wire6,uint32_t *ticks6,uint32_t *seq6)
{
    float motor[4];if(RLMotion31MapPolicyLegTorqueToMotorOrder(demand6,motor))return -1;
    clock_ms=tick;
    for(unsigned i=0;i<4;i++){dm[i]->ctrl.host_position_pd=0;dm[i]->ctrl.mode=0;
        dm[i]->ctrl.kp_set=0;dm[i]->ctrl.kd_set=0;DMMotorSetTorque(dm[i],motor[i]);}
    wheel_current[0]=demand6[5]*RL_MOTION31_WHEEL_CURRENT_PER_NM;
    wheel_current[1]=-demand6[2]*RL_MOTION31_WHEEL_CURRENT_PER_NM;
    MotorControlTask();task_calls++;
    memcpy(wire6,wire_model,sizeof(wire_model));memcpy(ticks6,wire_tick,sizeof(wire_tick));
    memcpy(seq6,wire_sequence,sizeof(wire_sequence));return critical_depth?-1:0;
}

/* Preserve distinct receipt ticks and the RobotTask snapshot tick. Old ABI
 * wrappers remain unchanged and continue to use a single declared sensor age. */
int motion_fw_observe_packet(void *context,uint32_t now,uint32_t snapshot_tick,uint32_t snapshot_sequence,
                            const uint32_t *receipt7,const float *native16,
                            const float *command3,float *out31)
{
    Bridge *b=context;if(!b)return -1;float length[2]={.28f,.28f},spread[2]={0};
    fill(&b->actor,snapshot_tick,receipt7[6],native16,command3,length,spread);
    memcpy(b->actor.feedback_receive_tick_ms,receipt7,6*sizeof(uint32_t));
    b->actor.sequence=snapshot_sequence;RLControlFillPrevious(&b->state,&b->actor);
    if(!RLControlFeedbackFresh(&b->actor,now))return -2;
    return RLMotion31BuildObservation(&b->actor.motion,out31);
}
int motion_fw_pd_packet(void *context,uint32_t now,const uint32_t *receipt7,
                       const float *native16,const float *command3,const float *length2,
                       const float *spread2,float *out30,uint32_t *meta8)
{
    /* Existing ABI prepares output layout; the real owner sees individual ages.
     * Temporarily construct the snapshot through a private helper is avoided:
     * use a local Bridge copy and the same output exporter after validation. */
    RLShadowSnapshot s;fill(&s,now,receipt7[6],native16,command3,length2,spread2);
    memcpy(s.feedback_receive_tick_ms,receipt7,6*sizeof(uint32_t));
    if(!RLControlFeedbackFresh(&s,now))return -2;
    return motion_fw_pd(context,now,receipt7[6],native16,command3,length2,spread2,out30,meta8);
}
