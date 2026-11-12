#include "rl_jump_platform.h"
#include "rl_jump_nominal.h"
#include <string.h>
volatile RLJumpLog g_rl_jump_log;
#if defined(__GNUC__)
#define J_WEAK __attribute__((weak))
#else
#define J_WEAK
#endif
J_WEAK int RLJumpModelInfo(RLJumpConfig *c) {
  (void)c;
  return -1;
}
J_WEAK int RLJumpModelInfer(const float input[37], float mean[6]) {
  (void)input;
  (void)mean;
  return -1;
}
J_WEAK int RLJumpPlatformSupport(RLJumpInput *input) {
  input->support_model_valid = 0;
  input->support_mask = 0;
  return -1;
}
void RLJumpPlatformInit(RLJumpPlatform *p, uint32_t now, RLJumpLockFn lock,
                        RLJumpLockFn unlock) {
  memset(p, 0, sizeof(*p));
  RLJumpInit(&p->state, now);
  RLJumpDefaultConfig(&p->config);
  (void)RLJumpModelInfo(&p->config);
  p->config.enabled =
      RL_JUMP_OUTPUT_ENABLED; /* a model cannot enable hardware output */
  p->lock = lock;
  p->unlock = unlock;
  p->ready = 1;
}
int RLJumpPlatformCycle(RLJumpPlatform *p, const RLJumpInput *i) {
  if (!p || !p->ready || !i)
    return 0;
  if (p->lock)
    p->lock();
  p->input = *i;
  (void)RLJumpStep(&p->state, i, &p->config);
  const int owns = RLJumpActive(&p->state) || p->state.owner == J_SAFE;
  if (!owns) {
    memset(&p->frame, 0, sizeof(p->frame));
    p->frame.owner = p->state.owner;
    p->frame.phase = p->state.phase;
    p->frame.fault = p->state.fault;
  }
  if (owns) {
    (void)RLJumpPd(&p->state, i, &p->frame);
    p->frame.owner = p->state.owner;
    p->frame.phase = p->state.phase;
    p->frame.fault = p->state.fault;
  }
  if (p->state.owner == J_SAFE) {
    memset(p->frame.final_motor_nm, 0, sizeof(p->frame.final_motor_nm));
    memset(p->frame.final_model_nm, 0, sizeof(p->frame.final_model_nm));
    memset(p->frame.wheel_current_raw, 0, sizeof(p->frame.wheel_current_raw));
  }
  const uint32_t serial = g_rl_jump_log.frame_end + 2u;
  g_rl_jump_log.frame_begin = serial - 1u;
  g_rl_jump_log.input_id = p->config.input_id;
  g_rl_jump_log.output_id = p->config.output_id;
  g_rl_jump_log.config_id = p->config.config_id;
  g_rl_jump_log.model_id = p->config.model_id;
  g_rl_jump_log.pd_executed = owns && p->state.owner == J_JUMP;
  g_rl_jump_log.Actor_snapshot_tick = p->Actor_snapshot_tick;
  g_rl_jump_log.Actor_start_tick = p->Actor_start_tick;
  g_rl_jump_log.Actor_end_tick = p->Actor_end_tick;
  g_rl_jump_log.state = p->state;
  g_rl_jump_log.feedback = *i;
  g_rl_jump_log.execution = p->frame;
  for (unsigned k = 0; k < 37; ++k)
    g_rl_jump_log.Actor_observation[k] = p->observation[k];
  g_rl_jump_log.frame_end = serial;
  g_rl_jump_log.frame_begin = serial;
  if (p->unlock)
    p->unlock();
  return owns;
}
int RLJumpPlatformActor(RLJumpPlatform *p, uint32_t now, RLJumpInferFn infer) {
  if (!p || !p->ready || !infer)
    return -1;
  RLJumpState s;
  RLJumpInput i;
  RLJumpConfig c;
  if (p->lock)
    p->lock();
  s = p->state;
  i = p->input;
  c = p->config;
  if (p->unlock)
    p->unlock();
  if (!RLJumpActive(&s) || !RLJumpIdentity(&c))
    return 1;
  i.now = now;
#if RL_JUMP_NOMINAL_V1
  RLJumpNominalUpdate(&s, now);
  s.last_tick = now;
#endif
  float input[37], mean[6];
  if (RLJumpEncode(&s, &i, input) || infer(input, mean)) {
    if (p->lock)
      p->lock();
    ++p->model_failure;
    if (p->unlock)
      p->unlock();
    return -1;
  }
  if (p->lock)
    p->lock();
  /* The hardware task supplies current monotonic time again at publication
     through its latest input. A stale phase/previous result is rejected. */
  i = p->input;
  if (p->clock)
    i.now = p->clock();
  p->Actor_snapshot_tick = s.last_tick;
  p->Actor_start_tick = now;
  p->Actor_end_tick = i.now;
  uint32_t seq = ++p->inference_sequence;
  if ((int32_t)(seq - p->state.adopted_sequence) <= 0)
    seq = p->inference_sequence = p->state.adopted_sequence + 1u;
  int rc = RLJumpPublishMean(&p->state, &i, &s, seq, mean);
  if (!rc) {
    memcpy(p->observation, input, sizeof(input));
    ++p->inference_count;
  }
  if (p->unlock)
    p->unlock();
  return rc;
}
