#include "rl_motion31.h"
#include "rl_policy_core.h"

#include <math.h>
#include <stddef.h>

#define RL_MOTION31_PI 3.14159265358979323846f
#define RL_MOTION31_TWO_PI (2.0f * RL_MOTION31_PI)
#define RL_MOTION31_HIP_INPUT_LIMIT_RAD 3.1f
#define RL_MOTION31_ACTIVE_INPUT_LIMIT_RAD 2.6f
/* Hardware angles start at backward horizontal. The model q=0 links are
   47.1033888 deg backward / 47.1080837 deg forward from downward vertical. */
#define RL_MOTION31_HIP_ZERO_RAD ((90.0f - 47.1033888f) * (RL_MOTION31_PI / 180.0f))
#define RL_MOTION31_ACTIVE_ZERO_RAD ((90.0f + 47.1080837f) * (RL_MOTION31_PI / 180.0f))

static int finite_array(const float *values, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

static float wrap_joint_angle(float angle_rad)
{
    float wrapped = remainderf(angle_rad, RL_MOTION31_TWO_PI);
    return wrapped >= RL_MOTION31_PI ? wrapped - RL_MOTION31_TWO_PI : wrapped;
}


/* Active links use the calibrated model branch [-2.6, 2.6]. Remove only
   equivalent encoder turns that return to that branch; do not fold a true
   out-of-range angle onto another mechanism branch. */
static float active_branch_angle(float angle_rad)
{
    if (fabsf(angle_rad) <= RL_MOTION31_ACTIVE_INPUT_LIMIT_RAD) return angle_rad;
    const float candidate = wrap_joint_angle(angle_rad);
    return fabsf(candidate) <= RL_MOTION31_ACTIVE_INPUT_LIMIT_RAD ? candidate : angle_rad;
}

void RLMotion31MapBodyImu(RLMotion31Snapshot *s,
                          const float gyro_flu_rad_s[3],
                          const float gravity_flu[3])
{
    if (s == NULL || gyro_flu_rad_s == NULL || gravity_flu == NULL) return;
    s->gyro_model_rad_s[0] = -gyro_flu_rad_s[1];
    s->gyro_model_rad_s[1] = gyro_flu_rad_s[2];
    s->gyro_model_rad_s[2] = -gyro_flu_rad_s[0];
    s->gravity_model[0] = -gravity_flu[1];
    s->gravity_model[1] = gravity_flu[2];
    s->gravity_model[2] = -gravity_flu[0];
}

void RLMotion31MapLqrAttitude(RLMotion31Snapshot *s,
                              float roll_rad, float pitch_rad,
                              float roll_rate_rad_s, float pitch_rate_rad_s,
                              float yaw_rate_rad_s)
{
    /* LQR theta_b=pitch is nose-down positive, and phi=yaw is left-turn
       positive. These are +FLU Y and +FLU Z rotations, respectively. */
    const float gyro_flu[3] = {roll_rate_rad_s, pitch_rate_rad_s, yaw_rate_rad_s};
    const float gravity_flu[3] = {
        sinf(pitch_rad),
        -sinf(roll_rad) * cosf(pitch_rad),
        -cosf(roll_rad) * cosf(pitch_rad)};
    RLMotion31MapBodyImu(s, gyro_flu, gravity_flu);
}

void RLMotion31MapJointAngles(RLMotion31Snapshot *s,
                              float right_hip_from_back_rad, float right_active_from_back_rad,
                              float left_hip_from_back_rad, float left_active_from_back_rad,
                              float right_hip_rate_rad_s, float right_active_rate_rad_s,
                              float left_hip_rate_rad_s, float left_active_rate_rad_s)
{
    if (s == NULL) return;
    /* This driver supplies bounded single-turn encoder angles plus fixed offsets.
       Allow the existing calibrated turn-offset tests, reject gross corrupted
       numbers before remainderf can turn them into a plausible small angle. */
    const float source[4]={right_hip_from_back_rad,right_active_from_back_rad,
                           left_hip_from_back_rad,left_active_from_back_rad};
    for (unsigned i=0;i<4u;++i) {
        if (!isfinite(source[i]) || fabsf(source[i])>32.0f*RL_MOTION31_PI) {
            for (unsigned j=0;j<4u;++j)s->joint_position_rad[j]=NAN;
            return;
        }
    }
    /* Physical L/R are viewed from behind. The model's left channels 0/1
       are on the physical right; its right channels 3/4 are on the left.
       All measured angles start at backward horizontal and increase CCW
       when viewed from the physical right toward the left. */
    /* Remove encoder turns after zero/direction conversion, before observation and PD. */
    s->joint_position_rad[0] = wrap_joint_angle(
        RL_MOTION31_HIP_ZERO_RAD - right_hip_from_back_rad);
    s->joint_position_rad[1] = active_branch_angle(
        right_active_from_back_rad - RL_MOTION31_ACTIVE_ZERO_RAD);
    s->joint_position_rad[2] = wrap_joint_angle(
        left_hip_from_back_rad - RL_MOTION31_HIP_ZERO_RAD);
    s->joint_position_rad[3] = active_branch_angle(
        left_active_from_back_rad - RL_MOTION31_ACTIVE_ZERO_RAD);
    s->joint_velocity_rad_s[0] = -right_hip_rate_rad_s;
    s->joint_velocity_rad_s[1] = right_active_rate_rad_s;
    s->joint_velocity_rad_s[3] = left_hip_rate_rad_s;
    s->joint_velocity_rad_s[4] = left_active_rate_rad_s;
}

void RLMotion31MapWheelMotorSpeeds(RLMotion31Snapshot *s,
                                   float left_motor_wheel_rad_s,
                                   float right_motor_wheel_rad_s)
{
    if (s == NULL) return;
    s->joint_velocity_rad_s[2] = -right_motor_wheel_rad_s;
    s->joint_velocity_rad_s[5] = left_motor_wheel_rad_s;
}

int RLMotion31BuildObservation(const RLMotion31Snapshot *s,
                               float o[RL_MOTION31_OBSERVATION_SIZE])
{
    if (o == NULL || RLMotion31ValidateSnapshot(s) != 0) return -1;
    const float command[3]={s->command_vx_m_s,s->command_yaw_rad_s,s->command_height_m};
    return RLPolicyEncode(s->gyro_model_rad_s,s->gravity_model,s->joint_position_rad,
        s->joint_velocity_rad_s,s->previous_target,command,0u,0u,o);
}

int RLMotion31ValidateSnapshot(const RLMotion31Snapshot *s)
{
    if (s == NULL || !s->previous_target_valid ||
        !finite_array(s->gyro_model_rad_s, 3u) ||
        !finite_array(s->gravity_model, 3u) ||
        !finite_array(s->joint_position_rad, 4u) ||
        !finite_array(s->joint_velocity_rad_s, 6u) ||
        !finite_array(s->previous_target, 6u) ||
        !isfinite(s->command_vx_m_s) || !isfinite(s->command_yaw_rad_s) ||
        !isfinite(s->command_height_m) ||
        s->command_height_m < RL_MOTION31_HEIGHT_MIN_M-1e-7f ||
        s->command_height_m > RL_MOTION31_HEIGHT_MAX_M+1e-7f) {
        return -1;
    }
    for (size_t i = 0; i < 4u; ++i) {
        const float limit = (i == 0u || i == 2u) ?
            RL_MOTION31_HIP_INPUT_LIMIT_RAD : RL_MOTION31_ACTIVE_INPUT_LIMIT_RAD;
        if (fabsf(s->joint_position_rad[i]) > limit) return -1;
    }
    /* Serialization/corruption guards, not motor capability or action clipping.
       These deliberately wide values reject huge finite garbage without claiming
       an identified gyro/motor operating envelope. */
    for (unsigned i=0;i<3u;++i) if(fabsf(s->gyro_model_rad_s[i])>10000.0f)return -1;
    for (unsigned i=0;i<6u;++i)
        if(fabsf(s->joint_velocity_rad_s[i])>1000000.0f ||
           fabsf(s->previous_target[i])>1000000.0f)return -1;
    const float norm2 = s->gravity_model[0]*s->gravity_model[0] +
                        s->gravity_model[1]*s->gravity_model[1] +
                        s->gravity_model[2]*s->gravity_model[2];
    /* Numerical unit-vector tolerance, not an accelerometer magnitude filter. */
    return fabsf(norm2-1.0f) <= 0.02f ? 0 : -1;
}

/* Global C0 quintic fit. Source six anchors are offline fitting evidence,
   not a runtime lookup table. x=(command_height-.35)/.05. */
int RLMotion31SupportCenter(double h,double out[4])
{
    if(out==NULL || !isfinite(h) || h<.30-1e-7 || h>.40+1e-7) return -1;
    const double x=(h-.35)/.05;
    const double L=h+.000024-.060;
    const double beta=acos((L*L+.21*.21-.25*.25)/(2*L*.21));
    const double bh=(((((-0.04464695144541761*x+(-0.0094959546615917666))*x+(0.070546704606128441))*x+(0.012914498339764401))*x+(-0.022292895866604146))*x+(-0.34856693858410187));
    const double ba=(((((0.016045303181803701*x+(0.0061990859752017189))*x+(-0.026839290520086146))*x+(-0.0084307569262746084))*x+(0.027428468197591423))*x+(-0.3586744187278344));
    const double hip=beta-47.1033888*(3.14159265358979323846/180.)+bh;
    const double active=beta-47.1080837*(3.14159265358979323846/180.)+ba;
    out[0]=hip;out[1]=active;out[2]=-hip;out[3]=active;
    return 0;
}

int RLMotion31DecodeOutput(const float action[RL_MOTION31_ACTION_SIZE],
                           float command_height_m,
                           float target[RL_MOTION31_ACTION_SIZE])
{
    const RLPolicyDecodeConfig config={.abi=RL_POLICY_CORE_ABI,
        .center_kind=RL_POLICY_CENTER_POLY5,.leg_scale=.15f,.wheel_scale=80.f,
        .height_min=.30,.height_max=.40};
    return RLPolicyDecode(&config,action,command_height_m,target);
}

int RLMotion31MapPolicyTargets(const float target[RL_MOTION31_ACTION_SIZE],
                               float hardware_joint_target_rad[4],
                               float motor_wheel_target_rad_s[2])
{
    if (target == NULL || hardware_joint_target_rad == NULL || motor_wheel_target_rad_s == NULL ||
        !finite_array(target, RL_MOTION31_ACTION_SIZE)) {
        return -1;
    }
    hardware_joint_target_rad[0] = RL_MOTION31_ACTIVE_ZERO_RAD + target[4]; /* lfangle */
    hardware_joint_target_rad[1] = RL_MOTION31_HIP_ZERO_RAD + target[3];    /* lbangle */
    hardware_joint_target_rad[2] = RL_MOTION31_ACTIVE_ZERO_RAD + target[1]; /* rfangle */
    hardware_joint_target_rad[3] = RL_MOTION31_HIP_ZERO_RAD - target[0];    /* rbangle */
    motor_wheel_target_rad_s[0] = target[5];  /* left motor: positive is backward */
    motor_wheel_target_rad_s[1] = -target[2]; /* right motor: positive is forward */
    return finite_array(hardware_joint_target_rad, 4u) &&
           finite_array(motor_wheel_target_rad_s, 2u) ? 0 : -1;
}

int RLMotion31ComputePdTorque(const RLMotion31Snapshot *s,
                              const float target[RL_MOTION31_ACTION_SIZE],
                              float requested[RL_MOTION31_ACTION_SIZE],
                              float limited[RL_MOTION31_ACTION_SIZE])
{
    if (!s) return -1;
    const float gains[3]={40.f,2.f,.2f},caps[6]={40.f,40.f,5.5f,40.f,40.f,5.5f};
    return RLPolicyPd(s->joint_position_rad,s->joint_velocity_rad_s,target,gains,caps,requested,limited);
}

int RLMotion31MapPolicyLegTorqueToMotorOrder(const float policy_torque_nm[RL_MOTION31_ACTION_SIZE],
                                            float leg_torque_motor_nm[4])
{
    if (policy_torque_nm == NULL || leg_torque_motor_nm == NULL ||
        !finite_array(policy_torque_nm, RL_MOTION31_ACTION_SIZE)) {
        return -1;
    }
    /* Existing DMMotor order/sign: lf, lb, rf, rb. */
    leg_torque_motor_nm[0] = -policy_torque_nm[4];
    leg_torque_motor_nm[1] = -policy_torque_nm[3];
    leg_torque_motor_nm[2] = policy_torque_nm[1];
    leg_torque_motor_nm[3] = -policy_torque_nm[0];
    return finite_array(leg_torque_motor_nm, 4u) ? 0 : -1;
}

int RLMotion31MapPolicyWheelTorqueToForwardOrder(const float policy_torque_nm[RL_MOTION31_ACTION_SIZE],
                                                 float wheel_torque_forward_nm[2])
{
    if (policy_torque_nm == NULL || wheel_torque_forward_nm == NULL ||
        !finite_array(policy_torque_nm, RL_MOTION31_ACTION_SIZE)) {
        return -1;
    }
    wheel_torque_forward_nm[0] = -policy_torque_nm[5]; /* physical left */
    wheel_torque_forward_nm[1] = -policy_torque_nm[2]; /* physical right */
    return finite_array(wheel_torque_forward_nm, 2u) ? 0 : -1;
}

int RLMotion31BuildPreview(const RLMotion31Snapshot *sample,
                           const float action[RL_MOTION31_ACTION_SIZE],
                           RLMotion31Preview *preview)
{
    if (preview == NULL ||
        RLMotion31BuildObservation(sample, preview->observation) != 0 ||
        RLMotion31DecodeOutput(action, sample->command_height_m, preview->target) != 0 ||
        RLMotion31MapPolicyTargets(preview->target, preview->hardware_joint_target_rad,
                                   preview->motor_wheel_target_rad_s) != 0) {
        return -1;
    }
    return 0;
}
