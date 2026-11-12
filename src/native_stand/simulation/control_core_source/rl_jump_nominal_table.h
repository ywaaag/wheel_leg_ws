/* Generated from SHA-bound trajectory.json; CPU nominal candidate, NOT hardware accepted. */
#ifndef RL_JUMP_NOMINAL_TABLE_H
#define RL_JUMP_NOMINAL_TABLE_H
#include <stdint.h>
#define RL_JUMP_TRAJECTORY_ID 0xa8b3b113u
#define RL_JUMP_TRAJECTORY_SHA "a8b3b11361314f59b4f755739cb5784f305f3eaa05104aad71ce4c0e65ff1a7c"
typedef struct {uint32_t intervals;float duration_ms;const float *knots_ms,*command,*geometry;} RLJumpTrajectory;
extern const RLJumpTrajectory rl_jump_trajectories[5];
#endif
