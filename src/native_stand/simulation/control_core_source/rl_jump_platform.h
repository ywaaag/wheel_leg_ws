#ifndef RL_JUMP_PLATFORM_H
#define RL_JUMP_PLATFORM_H
#include "rl_jump_core.h"
typedef int (*RLJumpInferFn)(const float input[37], float mean[6]);
typedef void (*RLJumpLockFn)(void);
typedef uint32_t (*RLJumpClockFn)(void);
typedef struct {
  RLJumpState state;
  RLJumpInput input;
  RLJumpConfig config;
  RLJumpFrame frame;
  uint32_t ready, inference_sequence, model_failure, inference_count;
  float observation[37];
  RLJumpLockFn lock, unlock;
  RLJumpClockFn clock;
  uint32_t Actor_snapshot_tick, Actor_start_tick, Actor_end_tick;
} RLJumpPlatform;
typedef struct {
  uint32_t frame_begin, frame_end, input_id, output_id, config_id, model_id;
  uint32_t pd_executed, Actor_snapshot_tick, Actor_start_tick, Actor_end_tick;
  RLJumpState state;
  RLJumpInput feedback;
  RLJumpFrame execution;
  float Actor_observation[37];
} RLJumpLog;
extern volatile RLJumpLog g_rl_jump_log;
void RLJumpPlatformInit(RLJumpPlatform *p, uint32_t now, RLJumpLockFn lock,
                        RLJumpLockFn unlock);
/* Chassis 5ms owner. Returns 1 when Jump or SAFE owns all six outputs. */
int RLJumpPlatformCycle(RLJumpPlatform *p, const RLJumpInput *input);
/* Existing 10ms Actor task; network runs outside critical section. */
int RLJumpPlatformActor(RLJumpPlatform *p, uint32_t now, RLJumpInferFn infer);
/* Intentionally unavailable until a real 37D model and identified support
   adapter are linked. These are NOT a zero-action/Motion31 substitute. */
int RLJumpModelInfo(RLJumpConfig *c);
int RLJumpModelInfer(const float input[37], float mean[6]);
int RLJumpPlatformSupport(RLJumpInput *input);
#endif
