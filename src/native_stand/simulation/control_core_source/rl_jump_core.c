#include "rl_jump_core.h"
#include "rl_jump_nominal.h"
#include <math.h>
#include <string.h>
static const unsigned legs[4] = {0, 1, 3, 4};
static int jump_finite_array(const float *x, unsigned n) {
  for (unsigned k = 0; k < n; ++k)
    if (!isfinite(x[k]))
      return 0;
  return 1;
}
static float clip(float x, float a) { return fminf(a, fmaxf(-a, x)); }
static uint32_t age(uint32_t now, uint32_t then) { return now - then; }
static void move(RLJumpState *s, uint32_t p, uint32_t now) {
  s->phase = p;
  s->phase_tick = now;
  ++s->phase_sequence;
  s->event_tick = now;
  ++s->event_id;
  s->stable_valid = 0;
  s->command_valid = 0;
}
static int fail(RLJumpState *s, uint32_t code, uint32_t now) {
  s->fault = code;
  s->owner = J_SAFE;
  move(s, J_FAULT, now);
  return -1;
}
static void snapshot(const RLJumpState *s, const RLJumpInput *i,
                     RLMotion31Snapshot *m) {
  memset(m, 0, sizeof(*m));
  memcpy(m->gyro_model_rad_s, i->gyro, sizeof(i->gyro));
  memcpy(m->gravity_model, i->gravity, sizeof(i->gravity));
  memcpy(m->joint_position_rad, i->q, sizeof(i->q));
  memcpy(m->joint_velocity_rad_s, i->dq, sizeof(i->dq));
  memcpy(m->previous_target, s->adopted, sizeof(s->adopted));
  m->previous_target_valid = (uint8_t)s->adopted_valid;
  m->command_vx_m_s = i->command[0];
  m->command_yaw_rad_s = i->command[1];
  m->command_height_m = i->command[2];
}
static int fresh(const RLJumpInput *i) {
  if (i->feedback_mask != 63u || !i->imu_valid ||
      age(i->now, i->sensor_tick) > 24u || age(i->now, i->imu_tick) > 24u)
    return 0;
  for (unsigned k = 0; k < 6; ++k)
    if (age(i->now, i->feedback_tick[k]) > 24u)
      return 0;
  return 1;
}
static int stationary(const RLJumpInput *i) {
  return jump_finite_array(i->command, 3) && fabsf(i->command[0]) <= .01f &&
         fabsf(i->command[1]) <= .01f && fabsf(i->command[2] - .34f) <= .005f;
}
static int entry(const RLJumpInput *i) {
  return i->motion_entry && i->gravity[1] < -.98f && fabsf(i->dq[2]) < 2.f &&
         fabsf(i->dq[5]) < 2.f &&
         fabsf(i->gyro[0]) + fabsf(i->gyro[1]) + fabsf(i->gyro[2]) < .3f;
}
void RLJumpDefaultConfig(RLJumpConfig *c) {
  memset(c, 0, sizeof(*c));
  c->abi = RL_JUMP_ABI;
  c->input_id = RL_JUMP_CONTRACT;
  c->output_id = RL_JUMP_OUTPUT;
  c->config_id = RL_JUMP_CONFIG;
  c->enabled = RL_JUMP_OUTPUT_ENABLED;
#if RL_JUMP_NOMINAL_V1
  c->trajectory_id = RL_JUMP_TRAJECTORY_ID;
#endif
}
void RLJumpInit(RLJumpState *s, uint32_t now) {
  memset(s, 0, sizeof(*s));
  s->abi = RL_JUMP_ABI;
  s->owner = J_MOTION;
  s->phase_tick = now;
  s->last_tick = now;
  s->candidate[0] = s->candidate[1] = 2u;
}
int RLJumpReset(RLJumpState *s, uint32_t now, int safe) {
  if (!safe)
    return -1;
  const uint32_t e = s->epoch + 1u, key = s->last_key, r = s->request_id;
  RLJumpInit(s, now);
  s->epoch = e;
  s->last_key = key;
  s->request_id = r;
  return 0;
}
int RLJumpIdentity(const RLJumpConfig *c) {
  return c && c->abi == RL_JUMP_ABI && c->input_id == RL_JUMP_CONTRACT &&
         c->output_id == RL_JUMP_OUTPUT && c->config_id == RL_JUMP_CONFIG &&
         c->enabled && c->model_available && c->model_id &&
         c->model_input_id == RL_JUMP_CONTRACT &&
         c->model_output_id == RL_JUMP_OUTPUT
#if RL_JUMP_NOMINAL_V1
         && c->trajectory_id == RL_JUMP_TRAJECTORY_ID
#endif
      ;
}
int RLJumpActive(const RLJumpState *s) {
  return s && s->phase >= J_CROUCH && s->phase <= J_SETTLE &&
         s->owner == J_JUMP;
}
int RLJumpAccelReliable(const float a[3]) {
  if (!jump_finite_array(a, 3))
    return 0;
  float n = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
  return n >= 47.155689f && n <= 162.635409f;
}
int RLJumpEstimateSupport(const RLJumpInput *i, float f[2], uint32_t valid[2]) {
  if (!i || !f || !valid)
    return -1;
  for (unsigned side = 0; side < 2; ++side) {
    valid[side] = 0;
    f[side] = 0;
    if (!i->support_model_valid || !(i->support_mask & (1u << side)) ||
        age(i->now, i->support_tick) > 20u)
      continue;
    const float *j = i->support_J + 4 * side;
    const float *t = i->driver_tau + 2 * side, *b = i->internal_tau + 2 * side;
    if (!jump_finite_array(j, 4) || !jump_finite_array(t, 2) ||
        !jump_finite_array(b, 2))
      continue;
    const float det = j[0] * j[3] - j[1] * j[2];
    if (fabsf(det) < 1e-4f)
      continue;
    f[side] = ((b[0] - t[0]) * j[3] - j[2] * (b[1] - t[1])) / det;
    if (isfinite(f[side]) && fabsf(f[side]) < 5000.f)
      valid[side] = 1;
  }
  return 0;
}
static void contacts(RLJumpState *s, const RLJumpInput *i) {
  uint32_t v[2];
  float f[2];
  RLJumpEstimateSupport(i, f, v);
  for (unsigned k = 0; k < 2; ++k) {
    s->support_N[k] = f[k];
    if (!v[k]) {
      s->contact_valid[k] = 0;
      s->candidate[k] = 2;
      s->off_valid[k] = 0;
      continue;
    }
    const uint32_t c =
        f[k] > 20.f
            ? 1u
            : (f[k] < 8.f ? 0u : (s->contact_valid[k] ? s->contact[k] : 2u));
    if (c != s->candidate[k]) {
      s->candidate[k] = c;
      s->candidate_tick[k] = i->now;
    }
    if (c < 2u && age(i->now, s->candidate_tick[k]) >= (c ? 5u : 8u)) {
      s->contact[k] = c;
      s->contact_valid[k] = 1;
    }
    if (s->contact_valid[k] && !s->contact[k]) {
      if (!s->off_valid[k])
        s->off_tick[k] = i->now;
      s->off_valid[k] = 1;
    } else
      s->off_valid[k] = 0;
  }
}
int RLJumpStep(RLJumpState *s, const RLJumpInput *i, const RLJumpConfig *c) {
  if (!s || !i || !c || s->abi != RL_JUMP_ABI)
    return -1;
  if (s->initialized &&
      (age(i->now, s->last_tick) == 0u || age(i->now, s->last_tick) > 10u))
    return fail(s, J_STALE_INPUT, i->now);
  s->last_tick = i->now;
  s->initialized = 1;
  const int rising = i->key_down && !s->last_key;
  s->last_key = !!i->key_down;
  if (s->phase == J_FAULT)
    return -1;
  if (!RLJumpActive(s))
    s->owner = i->upstream_owner;
  RLMotion31Snapshot m;
  snapshot(s, i, &m);
  /* Initialization from upstream is physical adoption, never shadow output. */
  if (!RLJumpActive(s) && i->adopted_valid &&
      jump_finite_array(i->upstream_target, 6)) {
    memcpy(s->adopted, i->upstream_target, sizeof(s->adopted));
    s->adopted_valid = 1;
    s->adopted_sequence = i->adopted_sequence;
  }
  snapshot(s, i, &m);
  const int good = fresh(i) && RLMotion31ValidateSnapshot(&m) == 0 &&
                   jump_finite_array(i->accel, 3);
  contacts(s, i);
  const int both = s->contact_valid[0] && s->contact_valid[1] &&
                   s->contact[0] && s->contact[1];
  const int known = s->contact_valid[0] && s->contact_valid[1];
  if (RLJumpActive(s) || s->phase == J_CONFIRM) {
    if (!i->enabled_mode)
      return fail(s, J_MODE_EXIT, i->now);
    if (i->legacy_jump || i->upstream_owner == J_RECOVERY)
      return fail(s, J_OWNER_CONFLICT, i->now);
    if (!RLJumpIdentity(c))
      return fail(s, J_BAD_IDENTITY, i->now);
    if (!stationary(i))
      return fail(s, J_BAD_INPUT, i->now);
    if (age(i->now, s->jump_tick) > 15000u)
      return fail(s, J_PHASE_TIMEOUT, i->now);
    if (!good)
      return fail(s, J_BAD_INPUT, i->now);
    if (!known)
      return fail(s, J_SUPPORT_UNKNOWN, i->now);
#if RL_JUMP_NOMINAL_V1
    for (unsigned k = 0; k < 2; ++k)
      if (!isfinite(i->leg_length[k]) || i->leg_length[k] <= .126f ||
          i->leg_length[k] >= .373f)
        return fail(s, J_MECHANICAL_RANGE, i->now);
#endif
  }
  if (rising) {
    ++s->request_id;
    if (s->phase != J_IDLE || i->upstream_owner != J_MOTION || i->legacy_jump)
      s->status = J_REJECT_BUSY;
    else if (!RLJumpIdentity(c))
      s->status = J_REJECT_MODEL;
    else if (!i->enabled_mode)
      s->status = J_REJECT_MODE;
    else if (!stationary(i))
      s->status = J_REJECT_COMMAND;
    else {
      s->status = J_WAIT;
      s->wait_tick = i->now;
    }
  }
  const uint32_t a = age(i->now, s->phase_tick);
  if (s->phase == J_IDLE) {
    const int eligible = good && both && entry(i) && stationary(i) &&
                         i->upstream_owner == J_MOTION && !i->legacy_jump;
    if (eligible) {
      if (!s->entry_valid)
        s->entry_tick = i->now;
      s->entry_valid = 1;
    } else
      s->entry_valid = 0;
    if (s->status == J_WAIT) {
      if (!i->enabled_mode)
        s->status = J_REJECT_MODE;
      else if (!stationary(i))
        s->status = J_REJECT_COMMAND;
      else if (!RLJumpIdentity(c))
        s->status = J_REJECT_MODEL;
      else if (i->upstream_owner != J_MOTION || i->legacy_jump)
        s->status = J_REJECT_BUSY;
      else if (age(i->now, s->wait_tick) > 500u)
        s->status = J_REJECT_TIMEOUT;
      else if (s->entry_valid && age(i->now, s->entry_tick) >= 100u) {
        ++s->epoch;
        s->owner = J_JUMP;
        s->status = J_ACCEPTED;
        s->first_hold = 1;
        s->airborne = s->success = 0;
        s->adopted_tick = i->now;
        s->jump_tick = i->now;
        move(s, J_CROUCH, i->now);
      }
    }
  } else if (s->phase == J_CROUCH) {
    if (!both)
      return fail(s, J_NO_TAKEOFF, i->now);
#if RL_JUMP_NOMINAL_V1
    if (a >= 350u && isfinite(i->leg_length[0]) && isfinite(i->leg_length[1]) &&
        i->leg_length[0] <= .15f && i->leg_length[1] <= .15f)
#else
    if (a >= 200u && (i->q[0] + i->q[1] - i->q[2] + i->q[3]) * .25f > .35f)
#endif
      move(s, J_PUSH, i->now);
    else if (a > 800u)
      return fail(s, J_PHASE_TIMEOUT, i->now);
  } else if (s->phase == J_PUSH || s->phase == J_FLIGHT || s->phase == J_LAND) {
    const int off0 = s->off_valid[0], off1 = s->off_valid[1];
    const float an = i->accel[0] * i->accel[0] + i->accel[1] * i->accel[1] +
                     i->accel[2] * i->accel[2];
    const int off = off0 && off1 && an < 9.f;
    if (s->phase == J_PUSH) {
      if (off) {
        s->airborne = 1;
        move(s, J_FLIGHT, i->now);
      } else if ((off0 && !off1 && age(i->now, s->off_tick[0]) > 30u) ||
                 (off1 && !off0 && age(i->now, s->off_tick[1]) > 30u))
        return fail(s, J_SINGLE_TAKEOFF, i->now);
      else if (a > 350u)
        return fail(s, J_NO_TAKEOFF, i->now);
    } else if (s->phase == J_FLIGHT) {
      if (s->contact[0] || s->contact[1])
        move(s, J_LAND, i->now);
      else if (a > 1200u)
        return fail(s, J_PHASE_TIMEOUT, i->now);
    } else {
      if (off)
        move(s, J_FLIGHT, i->now);
      else if (both && entry(i) && a >= 100u)
        move(s, J_SETTLE, i->now);
      else if (a > 800u)
        return fail(s, J_PHASE_TIMEOUT, i->now);
    }
  } else if (s->phase == J_SETTLE) {
    if (both && entry(i)) {
      if (!s->stable_valid)
        s->stable_tick = i->now;
      s->stable_valid = 1;
      if (age(i->now, s->stable_tick) >= 100u) {
        move(s, J_CONFIRM, i->now);
        s->owner = J_MOTION;
      }
    } else
      s->stable_valid = 0;
    if (a > 1500u)
      return fail(s, J_PHASE_TIMEOUT, i->now);
  } else if (s->phase == J_CONFIRM) {
    if (!both || !i->motion_success || i->upstream_owner != J_MOTION)
      return fail(s, J_CONFIRM_FAILED, i->now);
    if (a >= 10000u) {
      s->success = s->airborne;
      move(s, J_COOLDOWN, i->now);
    }
  } else if (s->phase == J_COOLDOWN && a >= 2000u) {
    s->status = J_NONE;
    s->entry_valid = 0;
    move(s, J_IDLE, i->now);
  }
#if RL_JUMP_NOMINAL_V1
  RLJumpNominalUpdate(s, i->now);
#endif
  return 0;
}
int RLJumpEncode(const RLJumpState *s, const RLJumpInput *i, float o[37]) {
  if (!s || !i || !o || !RLJumpActive(s) || !fresh(i))
    return -1;
  RLMotion31Snapshot m;
  snapshot(s, i, &m);
  if (RLMotion31BuildObservation(&m, o))
    return -1;
  const float task[6] = {0, 0, 1, 0, 0, 1};
  memcpy(o + 25, task, sizeof(task));
  for (unsigned k = 0; k < 5; ++k)
    o[31 + k] = (s->phase == k + 1u) ? 1.f : 0.f;
  o[36] = (float)age(i->now, s->phase_tick) * .001f;
  return 0;
}
int RLJumpDecodeMean(const float mean[6], float request[6], float protect[6]) {
#if RL_JUMP_NOMINAL_V1
  /* A nominal residual is meaningless without its phase/time snapshot. */
  (void)mean;
  (void)request;
  (void)protect;
  return -1;
#else

  if (!mean || !request || !protect || !jump_finite_array(mean, 6))
    return -1;
  for (unsigned k = 0; k < 6; ++k) {
    if (fabsf(mean[k]) > 1.000001f)
      return -1;
    const float scale =
        (k == 2u || k == 5u) ? RL_JUMP_WHEEL_SCALE : RL_JUMP_LEG_SCALE;
    request[k] = mean[k] * scale;
    protect[k] = clip(request[k], scale);
  }
  return 0;

#endif
}
int RLJumpPublishMean(RLJumpState *s, const RLJumpInput *i,
                      const RLJumpState *old, uint32_t seq,
                      const float mean[6]) {
  if (!s || !i || !old || !RLJumpActive(s) || old->owner != J_JUMP ||
      old->epoch != s->epoch || old->phase_sequence != s->phase_sequence ||
      old->adopted_sequence != s->adopted_sequence ||
      (int32_t)(seq - s->adopted_sequence) <= 0 ||
      (s->command_valid && (int32_t)(seq - s->command_sequence) <= 0) ||
      age(i->now, old->last_tick) > 24u)
    return -1;
  float r[6], p[6];
#if RL_JUMP_NOMINAL_V1
  if (RLJumpNominalDecode(old, mean, r, p))
#else
  if (RLJumpDecodeMean(mean, r, p))
#endif
    return -1;
#if RL_JUMP_NOMINAL_V1
  memcpy(s->sampled_nominal, old->nominal, sizeof(s->sampled_nominal));
#endif
  memcpy(s->mean, mean, sizeof(s->mean));
  memcpy(s->requested, r, sizeof(r));
  memcpy(s->protected_target, p, sizeof(p));
  memcpy(s->pending, p, sizeof(p));
  s->command_sequence = seq;
  s->command_tick = i->now;
  s->command_snapshot_tick = old->last_tick;
  s->command_epoch = s->epoch;
  s->command_phase_sequence = s->phase_sequence;
  s->command_previous_sequence = old->adopted_sequence;
  s->command_valid = 1;
  return 0;
}
static int motor_command(const RLJumpInput *i, const float target[6],
                         RLJumpFrame *f, unsigned final_only) {
  if (!i || !target || !f)
    return -1;
  RLJumpState temporary = {0};
  temporary.adopted_valid = 1;
  memcpy(temporary.adopted, target, sizeof(temporary.adopted));
  RLMotion31Snapshot m;
  snapshot(&temporary, i, &m);
  if (RLMotion31ValidateSnapshot(&m) ||
      RLMotion31ComputePdTorque(&m, target, f->pd_nm, f->limited_nm))
    return -1;
  /* Existing PD helper has an intermediate 5.5Nm wheel cap. The physical
     driver cap below is the final 8192 raw equivalent (2.44694786Nm). */
  float motor[4], zero[4] = {0};
  /* New explicit sum-before-cap contract. PD gains and final40Nm unchanged.
     This is Jump-only: upstream Motion retains its original double cap. */
  const float *leg_pd = final_only ? f->pd_nm : f->limited_nm;
  if (RLMotion31MapPolicyLegTorqueToMotorOrder(leg_pd, motor) ||
      RLControlCompensateSpring(zero, i->leg_length, i->leg_spread,
                                f->compensation_motor_nm) ||
      RLControlCompensateSpring(motor, i->leg_length, i->leg_spread,
                                f->final_motor_nm))
    return -1;
  for (unsigned k = 0; k < 6; ++k) {
    if (k == 2u || k == 5u) {
      f->p_nm[k] = f->pd_nm[k];
      f->d_nm[k] = 0;
      f->limited_nm[k] = clip(f->limited_nm[k], RL_JUMP_WHEEL_CAP);
    } else {
      unsigned j = k < 2u ? k : k - 1u;
      f->p_nm[k] = 40.f * (target[k] - i->q[j]);
      f->d_nm[k] = -2.f * i->dq[k];
    }
  }
  f->final_model_nm[0] = -f->final_motor_nm[3];
  f->final_model_nm[1] = f->final_motor_nm[2];
  f->final_model_nm[3] = -f->final_motor_nm[1];
  f->final_model_nm[4] = -f->final_motor_nm[0];
  f->final_model_nm[2] = f->limited_nm[2];
  f->final_model_nm[5] = f->limited_nm[5];
  f->wheel_current_raw[0] =
      clip(f->final_model_nm[5] * RL_MOTION31_WHEEL_CURRENT_PER_NM, 8192.f);
  f->wheel_current_raw[1] =
      clip(-f->final_model_nm[2] * RL_MOTION31_WHEEL_CURRENT_PER_NM, 8192.f);
  memcpy(f->adopted, target, sizeof(f->adopted));
  return 0;
}
int RLJumpMotorCommand(const RLJumpInput *i, const float target[6],
                       RLJumpFrame *f) {
  return motor_command(i, target, f, RL_JUMP_NOMINAL_V1);
}
int RLJumpUpstreamMotorCommand(const RLJumpInput *i, const float target[6],
                               RLJumpFrame *f) {
  return motor_command(i, target, f, 0u);
}
int RLJumpPd(RLJumpState *s, const RLJumpInput *i, RLJumpFrame *f) {
  if (!s || !i || !f)
    return -1;
  memset(f, 0, sizeof(*f));
  f->owner = s->owner;
  f->phase = s->phase;
  f->fault = s->fault;
  if (!RLJumpActive(s))
    return s->owner == J_SAFE ? -1 : 1;
  if (!fresh(i))
    return fail(s, J_STALE_INPUT, i->now);
  RLJumpState next = *s;
  if (next.first_hold)
    next.first_hold = 0;
  else if (next.command_valid && age(i->now, next.command_tick) >= 5u &&
           age(i->now, next.command_snapshot_tick) <= 24u &&
           next.command_epoch == next.epoch &&
           next.command_phase_sequence == next.phase_sequence &&
           next.command_previous_sequence == next.adopted_sequence) {
    memcpy(next.adopted, next.pending, sizeof(next.adopted));
    next.adopted_sequence = next.command_sequence;
    next.adopted_tick = i->now;
    next.command_valid = 0;
  }
  if (age(i->now, next.adopted_tick) > 24u)
    return fail(s, J_STALE_TARGET, i->now);
  if (RLJumpMotorCommand(i, next.adopted, f))
    return fail(s, J_BAD_PD, i->now);
  memcpy(f->requested, next.requested, sizeof(f->requested));
  memcpy(f->protected_target, next.protected_target,
         sizeof(f->protected_target));
#if RL_JUMP_NOMINAL_V1
  memcpy(f->nominal, next.sampled_nominal, sizeof(f->nominal));
  for (unsigned j = 0; j < 6; ++j)
    f->residual[j] = next.requested[j] - next.sampled_nominal[j];
#endif
  f->adopted_sequence = next.adopted_sequence;
  f->adopted_tick = next.adopted_tick;
  for (unsigned j = 0; j < 4; ++j) {
    unsigned k = legs[j];
    if (f->requested[k] != f->protected_target[k])
      f->clipped_mask |= 1u << k;
  }
  *s = next;
  return 0;
}
int RLJumpBatch(RLJumpState *s, const RLJumpInput *i, const RLJumpConfig *c,
                RLJumpFrame *f, uint32_t n, int pd) {
  int errors = 0;
  for (uint32_t k = 0; k < n; ++k) {
    int rc = RLJumpStep(s + k, i + k, c);
    if (pd && RLJumpPd(s + k, i + k, f + k) < 0)
      rc = -1;
    if (rc < 0)
      ++errors;
  }
  return errors;
}
uint32_t RLJumpSize(uint32_t which) {
  switch (which) {
  case 0:
    return RL_JUMP_ABI;
  case 1:
    return sizeof(RLJumpConfig);
  case 2:
    return sizeof(RLJumpInput);
  case 3:
    return sizeof(RLJumpState);
  case 4:
    return sizeof(RLJumpFrame);
  default:
    return 0;
  }
}
void RLJumpEncodeBatch(const RLJumpState *s, const RLJumpInput *i, float *o,
                       uint32_t *v, uint32_t n) {
  for (uint32_t k = 0; k < n; ++k) {
    memset(o + 37 * k, 0, 37 * sizeof(float));
    v[k] = RLJumpEncode(s + k, i + k, o + 37 * k) == 0;
  }
}
void RLJumpPublishBatch(RLJumpState *s, const RLJumpInput *i,
                        const RLJumpState *old, const uint32_t *seq,
                        const float *mean, const uint32_t *mask, int32_t *rc,
                        uint32_t n) {
  for (uint32_t k = 0; k < n; ++k)
    rc[k] = mask[k]
                ? RLJumpPublishMean(s + k, i + k, old + k, seq[k], mean + 6 * k)
                : 1;
}
void RLJumpPdBatch(RLJumpState *s, const RLJumpInput *i, RLJumpFrame *f,
                   int32_t *rc, uint32_t n) {
  for (uint32_t k = 0; k < n; ++k)
    rc[k] = RLJumpPd(s + k, i + k, f + k);
}
