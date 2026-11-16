#ifndef RL_JUMP_CORE_H
#define RL_JUMP_CORE_H
#include "rl_control.h"
#include <stddef.h>
#include <stdint.h>
#ifndef RL_JUMP_NOMINAL_V1
#define RL_JUMP_NOMINAL_V1 0u
#endif
#ifndef RL_JUMP_DIRECT_V1
#define RL_JUMP_DIRECT_V1 1u
#endif
#if RL_JUMP_DIRECT_V1 && RL_JUMP_NOMINAL_V1
#error "direct raw and nominal residual outputs cannot be combined"
#endif
#define RL_JUMP_CONTRACT 0x4a370102u
#if RL_JUMP_DIRECT_V1
#define RL_JUMP_ABI 4u
#define RL_JUMP_OUTPUT 0x4a370401u
#define RL_JUMP_CONFIG 0x4a050401u
#elif RL_JUMP_NOMINAL_V1
#define RL_JUMP_ABI 2u
#define RL_JUMP_OUTPUT 0x4a370201u
#define RL_JUMP_CONFIG 0x4a050201u
#else
#define RL_JUMP_ABI 1u
#define RL_JUMP_OUTPUT 0x4a150101u
#define RL_JUMP_CONFIG 0x4a050101u
#endif
#define RL_JUMP_OUTCOME_CONFIG 0x4a050402u
#define RL_JUMP_WHEEL_CAP 4.8f
#define RL_JUMP_WHEEL_RAW_CAP (RL_JUMP_WHEEL_CAP * RL_MOTION31_WHEEL_CURRENT_PER_NM)
/* Absolute requested position range is NOT a reachable configuration box. */
#define RL_JUMP_LEG_SCALE 1.5f
#define RL_JUMP_WHEEL_SCALE 80.0f
/* Raw policy gains, NOT action bounds. Physical target caps remain separate. */
#define RL_JUMP_DIRECT_LEG_GAIN .5f
#define RL_JUMP_DIRECT_WHEEL_GAIN 10.0f
#ifndef RL_JUMP_OUTPUT_ENABLED
#define RL_JUMP_OUTPUT_ENABLED 0u
#endif
enum {
  J_IDLE,
  J_CROUCH,
  J_PUSH,
  J_FLIGHT,
  J_LAND,
  J_SETTLE,
  J_CONFIRM,
  J_COOLDOWN,
  J_FAULT,
  J_BALANCE
};
enum { J_LEGACY, J_MOTION, J_JUMP, J_RECOVERY, J_SAFE };
enum {
  J_NONE,
  J_WAIT,
  J_ACCEPTED,
  J_REJECT_BUSY,
  J_REJECT_COMMAND,
  J_REJECT_MODEL,
  J_REJECT_MODE,
  J_REJECT_TIMEOUT
};
enum {
  J_OK,
  J_BAD_INPUT,
  J_STALE_INPUT,
  J_OWNER_CONFLICT,
  J_MODE_EXIT,
  J_SUPPORT_UNKNOWN,
  J_NO_TAKEOFF,
  J_SINGLE_TAKEOFF,
  J_PHASE_TIMEOUT,
  J_STALE_TARGET,
  J_BAD_IDENTITY,
  J_BAD_PD,
  J_CONFIRM_FAILED,
  J_MECHANICAL_RANGE
};
typedef struct {
  uint32_t abi, input_id, output_id, config_id, enabled;
  uint32_t model_available, model_id, model_input_id, model_output_id;
#if RL_JUMP_NOMINAL_V1
  uint32_t trajectory_id;
#endif
} RLJumpConfig;
/* All sensor ticks are receiver-local timestamps, not acquisition claims.
   J is per side [d foot_height/d hip, d foot_height/d active;
                   d foot_forward/d hip, d foot_forward/d active], model order.
   tau_internal - driver = J^T * [normal_N, tangential_N]. Internal demand
   explicitly includes gravity, spring, inertia and friction. No LQR request.
   An unavailable internal-load model MUST leave support_model_valid=0. */
typedef struct {
  uint32_t now, sensor_tick, imu_tick, feedback_mask, imu_valid;
  uint32_t feedback_tick[6];
  uint32_t enabled_mode, key_down, upstream_owner, legacy_jump;
  uint32_t motion_entry, motion_success, support_tick, support_mask;
  uint32_t support_model_valid, adopted_valid, adopted_sequence;
  float gyro[3], gravity[3], q[4], dq[6], command[3];
  float upstream_target[6], accel[3], driver_tau[4], internal_tau[4];
  float support_J[8], leg_length[2], leg_spread[2];
} RLJumpInput;
/* No pointers/padding-dependent bools, allocation or RTOS objects. Raw state
   is serialized with ABI, contract, endian, sizes and SHA envelope by caller.
 */
typedef struct {
  uint32_t abi, phase, owner, fault, status, epoch, phase_tick, last_tick,
      jump_tick;
  uint32_t initialized, last_key, request_id, wait_tick, entry_tick,
      entry_valid;
  uint32_t phase_sequence, adopted_sequence, adopted_tick, adopted_valid;
  uint32_t command_sequence, command_tick, command_snapshot_tick, command_epoch;
  uint32_t command_phase_sequence, command_previous_sequence, command_valid;
  uint32_t first_hold, airborne, success, stable_tick, stable_valid;
  uint32_t contact[2], candidate[2], candidate_tick[2], contact_valid[2];
  uint32_t off_tick[2], off_valid[2], event_tick, event_id;
  float adopted[6], mean[6], requested[6], protected_target[6], pending[6];
  float support_N[2];
#if RL_JUMP_NOMINAL_V1
  uint32_t reference_initialized, reference_phase_sequence, reference_tick;
  float nominal[6], nominal_rate[6], nominal_acc[6], geometric[6];
  float bridge_delta[6], bridge_rate[6], bridge_acc[6], sampled_nominal[6];
#endif
} RLJumpState;
typedef struct {
  uint32_t owner, phase, fault, adopted_sequence, adopted_tick, clipped_mask;
  float requested[6], protected_target[6], adopted[6];
  float p_nm[6], d_nm[6], pd_nm[6], limited_nm[6];
  float compensation_motor_nm[4], final_motor_nm[4], wheel_current_raw[2];
  float final_model_nm[6];
#if RL_JUMP_NOMINAL_V1
  float nominal[6], residual[6];
#endif
} RLJumpFrame;
void RLJumpDefaultConfig(RLJumpConfig *c);
void RLJumpInit(RLJumpState *s, uint32_t now);
int RLJumpReset(RLJumpState *s, uint32_t now, int safe_mode);
int RLJumpIdentity(const RLJumpConfig *c);
int RLJumpActive(const RLJumpState *s);
int RLJumpStep(RLJumpState *s, const RLJumpInput *i, const RLJumpConfig *c);
int RLJumpEncode(const RLJumpState *s, const RLJumpInput *i, float out[37]);
int RLJumpDecodeMean(const float mean[6], float requested[6],
                     float protected_target[6]);
int RLJumpPublishMean(RLJumpState *s, const RLJumpInput *i,
                      const RLJumpState *snapshot, uint32_t sequence,
                      const float mean[6]);
int RLJumpMotorCommand(const RLJumpInput *i, const float target[6],
                       RLJumpFrame *f);
/* Existing non-Jump owner contract, including the original inner PD cap. */
int RLJumpUpstreamMotorCommand(const RLJumpInput *i, const float target[6],
                               RLJumpFrame *f);
int RLJumpPd(RLJumpState *s, const RLJumpInput *i, RLJumpFrame *f);
int RLJumpEstimateSupport(const RLJumpInput *i, float support[2],
                          uint32_t valid[2]);
int RLJumpAccelReliable(const float accel[3]);
/* One ABI crossing per batch, the exact scalar firmware C functions. */
int RLJumpBatch(RLJumpState *s, const RLJumpInput *i, const RLJumpConfig *c,
                RLJumpFrame *f, uint32_t n, int run_pd);
uint32_t RLJumpSize(uint32_t which);
void RLJumpEncodeBatch(const RLJumpState *s, const RLJumpInput *i, float *out,
                       uint32_t *valid, uint32_t n);
void RLJumpPublishBatch(RLJumpState *s, const RLJumpInput *i,
                        const RLJumpState *snapshot, const uint32_t *sequence,
                        const float *mean, const uint32_t *mask, int32_t *rc,
                        uint32_t n);
void RLJumpPdBatch(RLJumpState *s, const RLJumpInput *i, RLJumpFrame *f,
                   int32_t *rc, uint32_t n);
#endif
