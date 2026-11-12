#ifndef RL_JUMP_NOMINAL_H
#define RL_JUMP_NOMINAL_H
#include "rl_jump_core.h"
#if RL_JUMP_NOMINAL_V1
#include "rl_jump_nominal_table.h"
void RLJumpNominalUpdate(RLJumpState *s, uint32_t now);
int RLJumpNominalDecode(const RLJumpState *s, const float mean[6],
                        float request[6], float protect[6]);
int RLJumpNominalSample(uint32_t phase, uint32_t elapsed_ms, float nominal[6],
                        float geometric[6], float rate[6], float acc[6]);
void RLJumpNominalObserveBatch(const RLJumpState *s, const RLJumpInput *i,
                               RLJumpState *snapshot, float *out,
                               uint32_t *valid, uint32_t n);
#endif
#endif
