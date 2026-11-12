#include "rl_jump_nominal.h"
#if RL_JUMP_NOMINAL_V1
#include <math.h>
#include <string.h>
static const float residual_scale[5][6] = {
    {.10f, .10f, 10.f, .10f, .10f, 10.f},
    {.05f, .05f, 10.f, .05f, .05f, 10.f},
    {.10f, .10f, 10.f, .10f, .10f, 10.f},
    {.10f, .10f, 10.f, .10f, .10f, 10.f},
    {.10f, .10f, 10.f, .10f, .10f, 10.f}};
static float cap(float x, float a) { return fminf(a, fmaxf(-a, x)); }
int RLJumpNominalSample(uint32_t phase, uint32_t elapsed, float n[6],
                        float g[6], float v[6], float a[6]) {
  if (phase < J_CROUCH || phase > J_SETTLE || !n || !g || !v || !a)
    return -1;
  const RLJumpTrajectory *t = &rl_jump_trajectories[phase - 1];
  const float ms = fminf((float)elapsed, t->duration_ms);
  uint32_t k = 0;
  while (k + 1 < t->intervals && ms >= t->knots_ms[k + 1])
    ++k;
  const float dt = (ms - t->knots_ms[k]) * .001f;
  for (unsigned j = 0; j < 6; ++j) {
    const float *c = t->command + k * 24 + j * 4,
                *d = t->geometry + k * 24 + j * 4;
    n[j] = ((c[0] * dt + c[1]) * dt + c[2]) * dt + c[3];
    g[j] = ((d[0] * dt + d[1]) * dt + d[2]) * dt + d[3];
    v[j] = elapsed >= t->duration_ms ? 0.f
                                     : (3 * c[0] * dt + 2 * c[1]) * dt + c[2];
    a[j] = elapsed >= t->duration_ms ? 0.f : 6 * c[0] * dt + 2 * c[1];
  }
  return 0;
}
void RLJumpNominalUpdate(RLJumpState *s, uint32_t now) {
  if (!RLJumpActive(s))
    return;
  const uint32_t elapsed = now - s->phase_tick;
  float n[6], g[6], v[6], a[6];
  if (RLJumpNominalSample(s->phase, elapsed, n, g, v, a))
    return;
  if (!s->reference_initialized ||
      s->reference_phase_sequence != s->phase_sequence) {
    float n0[6], g0[6], v0[6], a0[6];
    RLJumpNominalSample(s->phase, 0, n0, g0, v0, a0);
    for (unsigned j = 0; j < 6; ++j) {
      const float anchor =
          s->adopted[j]; /* every stage starts at actual adopted target */
      s->bridge_delta[j] = anchor - n0[j];
      s->bridge_rate[j] =
          (s->reference_initialized ? s->nominal_rate[j] : 0.f) - v0[j];
      s->bridge_acc[j] =
          (s->reference_initialized ? s->nominal_acc[j] : 0.f) - a0[j];
    }
    s->reference_initialized = 1;
    s->reference_phase_sequence = s->phase_sequence;
  }
  /* Explicit40ms C2 handover of nominal command, NOT Actor feedback filtering.
     The deployed Actor samples this reference10ms; C adopts targets5ms later.
   */
  const float T = .04f, x = fminf((float)elapsed / 40.f, 1.f);
  for (unsigned j = 0; j < 6; ++j) {
    float d = s->bridge_delta[j], r = s->bridge_rate[j] * T,
          b = s->bridge_acc[j] * T * T;
    float c3 = -10 * d - 6 * r - 1.5f * b, c4 = 15 * d + 8 * r + 1.5f * b,
          c5 = -6 * d - 3 * r - .5f * b;
    float y = d + r * x + .5f * b * x * x + c3 * x * x * x +
              c4 * x * x * x * x + c5 * x * x * x * x * x;
    float dy = r + b * x + 3 * c3 * x * x + 4 * c4 * x * x * x +
               5 * c5 * x * x * x * x;
    float ddy = b + 6 * c3 * x + 12 * c4 * x * x + 20 * c5 * x * x * x;
    s->nominal[j] = n[j] + (elapsed < 40u ? y : 0.f);
    s->nominal_rate[j] = v[j] + (elapsed < 40u ? dy / T : 0.f);
    s->nominal_acc[j] = a[j] + (elapsed < 40u ? ddy / (T * T) : 0.f);
    s->geometric[j] = g[j];
  }
  s->reference_tick = now;
}
int RLJumpNominalDecode(const RLJumpState *s, const float mean[6],
                        float request[6], float protect[6]) {
  if (!RLJumpActive(s) || !s->reference_initialized)
    return -1;
  const float activation =
      fminf((float)(s->reference_tick - s->phase_tick) / 40.f, 1.f);
  for (unsigned j = 0; j < 6; ++j) {
    if (!isfinite(mean[j]) || fabsf(mean[j]) > 1.000001f ||
        !isfinite(s->nominal[j]))
      return -1;
    request[j] =
        s->nominal[j] + activation * residual_scale[s->phase - 1][j] * mean[j];
    protect[j] = cap(request[j], j == 2u || j == 5u ? 80.f : 1.5f);
  }
  return 0;
}
void RLJumpNominalObserveBatch(const RLJumpState *s, const RLJumpInput *i,
                               RLJumpState *snap, float *out, uint32_t *valid,
                               uint32_t n) {
  for (uint32_t k = 0; k < n; ++k) {
    snap[k] = s[k];
    RLJumpNominalUpdate(snap + k, i[k].now);
    snap[k].last_tick = i[k].now;
    memset(out + 37 * k, 0, 37 * sizeof(float));
    valid[k] = RLJumpEncode(snap + k, i + k, out + 37 * k) == 0;
  }
}
#endif
