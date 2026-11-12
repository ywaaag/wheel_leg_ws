#!/usr/bin/env bash
# Reusable read-only MuJoCo preview for Motion31 checkpoints (desktop :2 by default).
#
#   bash scripts/play_motion31_mujoco.sh \
#     --checkpoint runs/<run>/formal/model_update_500.pt
#
# Defaults are the browser panel: --viewer is added, and --panel on the LAN address with the
# session caps failsafe. Pass --keyboard for window keys, --commands ... for scored playback,
# or PLAY_HEADLESS=1 for no window at all; an explicit flag always wins.
#
# Several checkpoints may be listed after --checkpoint; a read-only preview plays them in turn.
# --spec is derived from runs/<run>/... as configs/experiments/<run>/spec.json, from an ONNX
# manifest's parent_spec, or from the spec_sha256 a checkpoint stores; it is only needed to
# override that identity. The execution
# chain is chosen by that spec's own version: mcu_execution.profile plays the post-PD MCU chain
# with the training-fixed MCU leg armature, legacy specs keep motion_timing.configuration and its
# declared armature level. Neither is guessed from training progress. Explicit flags win.
#
# Environment: PLAY_DISPLAY (default :2), PLAY_PANEL_HOST (192.168.250.1), PLAY_PANEL_PORT (8766),
# PLAY_HEADLESS=1, PLAY_PANEL=0 (turn the web panel off), PY (interpreter), PLAY_DRY_RUN=1 to print only.
# The panel port advances to the next free one when the requested port is already taken, so an
# earlier preview that is still open cannot silently leave a new playback without web control.
# Omitting --output-dir keeps the preview read-only: no manifest, trace or terminal report.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
# 解释器由调用环境显式选择（PY）；未指定时从环境中找可用解释器。
if [ -z "${PY:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then PY=python3; elif command -v python >/dev/null 2>&1; then PY=python; else echo "play: 未找到可用解释器，请设置 PY" >&2; exit 2; fi
fi
PANEL_HOST=${PLAY_PANEL_HOST:-192.168.250.1}
PANEL_PORT=${PLAY_PANEL_PORT:-8766}
export DISPLAY=${PLAY_DISPLAY:-${DISPLAY:-:2}}
export XAUTHORITY=${XAUTHORITY:-/home/kellen/.Xauthority}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Web control is the default for every interactive playback, so never let a busy port
# silently cost the panel: fall back to the next free port and print the real URL.
if [ "${PLAY_PANEL:-1}" != "0" ] && [ "${PLAY_HEADLESS:-0}" != "1" ]; then
  resolved=$("$PY" - "$PANEL_HOST" "$PANEL_PORT" <<'PY'
import socket, sys
host, start = sys.argv[1], int(sys.argv[2])
def free(address, port):
    probe = socket.socket()
    try:
        probe.bind((address, port)); return True
    except OSError:
        return False
    finally:
        probe.close()
if not free(host, 0):
    print(f'play: panel host {host} is not local; binding 0.0.0.0 instead', file=sys.stderr)
    host = '0.0.0.0'
for port in range(start, start + 21):
    if free(host, port):
        print(host, port); break
else:
    sys.exit(f'play: no free panel port in {start}..{start + 20}')
PY
  ) || exit 2
  read -r PANEL_HOST PANEL_PORT <<<"$resolved"
  if [ "$PANEL_HOST" = "0.0.0.0" ]; then
    echo "play: web panel http://127.0.0.1:${PANEL_PORT}/ (bound 0.0.0.0, also reachable on this host's LAN address)" >&2
  else
    echo "play: web panel http://${PANEL_HOST}:${PANEL_PORT}/" >&2
  fi
fi

echo "play: start height defaults to .34 m (standing pose); --init-height-m 0 keeps the recipe origin start" >&2

usage() {
  cat >&2 <<'TXT'
usage: bash scripts/play_motion31_mujoco.sh --checkpoint <model_update_N.pt> [more checkpoints]
   or: bash scripts/play_motion31_mujoco.sh --models big=<ckpt.pt> small=<actor.onnx> [more NAME=PATH]
   or: bash scripts/play_motion31_mujoco.sh --replay-dir <completed diagnostic clips> [--clip NAME]
         [--start-paused] [--playback-speed .25] [--wall-seconds 600]
         [--spec <spec.json>] [--leg-armature-m <level>] [--viewer] [--keyboard]
         [--panel --panel-host <ip> --panel-port <port>] [--commands vx ...] [--yaw-commands w ...]
         [--max-speed mps] [--max-yaw radps] [--wall-seconds s] [--zero-latency]
       --models loads several models into one plant so the panel can switch them live
       (an ONNX entry needs its export manifest.json beside it, or --onnx-manifest).
       Default control is the browser panel (--viewer --panel on the LAN address, caps -.5/.5 m/s and yaw);
       the URL and the effective port are printed before playback starts, and a taken port moves
       to the next free one. --keyboard, --commands or PLAY_HEADLESS=1 opt out of that default;
       PLAY_PANEL=0 keeps the window but never opens the panel.
       --spec is optional: runs/<run>/formal/*.pt resolves to configs/experiments/<run>/spec.json,
       an ONNX package uses its manifest parent_spec (SHA-checked), and any other checkpoint is
       matched by the spec_sha256 it stores; pass --spec only to override that identity.
       --leg-armature-m defaults to the spec-declared motion_timing.configuration level.
      --leg-response defaults to bilateral_modal_v1 for Motion31 MCU playback: the delivered
      bilateral handoff (artifacts/2026_10_30__bilateral_leg_response_handoff) replaces the
      per-joint leg transport/first-order response with per-side modal ones; PD40/2, +-40Nm and
      the plant itself do not change. PLAY_LEG_RESPONSE=spec restores the spec chain.
       No --output-dir means read-only preview: nothing is written.
TXT
}

args=("$@")
for argument in "$@"; do
  if [ "$argument" = "--replay-dir" ]; then
    replay_panel=(--panel --panel-host "$PANEL_HOST" --panel-port "$PANEL_PORT")
    if [ "${PLAY_PANEL:-1}" = "0" ]; then replay_panel=(); fi
    replay_cmd=("$PY" scripts/play_motion_handover.py --viewer "${replay_panel[@]+"${replay_panel[@]}"}" "${args[@]}")
    echo "play: stored diagnostic trajectory replay (no live Actor or physics): ${replay_cmd[*]}" >&2
    if [ "${PLAY_DRY_RUN:-0}" = "1" ]; then exit 0; fi
    exec "${replay_cmd[@]}"
  fi
done
has_spec=0
has_armature=0
has_leg_response=0
has_plant_profile=0
has_viewer=0
has_panel=0
has_keyboard=0
has_commands=0
has_max_speed=0
has_max_yaw=0
first_checkpoint=""
model_paths=()
spec_path=""
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    --models)
      j=$((i + 1))
      while [ $j -lt ${#args[@]} ] && [ "${args[j]#-}" = "${args[j]}" ]; do
        entry=${args[j]}
        model_paths+=("${entry#*=}")
        j=$((j + 1))
      done
      i=$((j - 1))
      ;;
    --spec)
      has_spec=1
      if [ $((i + 1)) -lt ${#args[@]} ]; then spec_path=${args[i + 1]}; fi
      ;;
    --leg-armature-m) has_armature=1 ;;
    --leg-response) has_leg_response=1 ;;
    --plant-profile) has_plant_profile=1 ;;
    --viewer) has_viewer=1 ;;
    --panel) has_panel=1 ;;
    --keyboard) has_keyboard=1 ;;
    --commands) has_commands=1 ;;
    --max-speed) has_max_speed=1 ;;
    --max-yaw) has_max_yaw=1 ;;
    --checkpoint)
      if [ -z "$first_checkpoint" ] && [ $((i + 1)) -lt ${#args[@]} ]; then
        first_checkpoint=${args[i + 1]}
      fi
      ;;
    -h | --help)
      usage
      exit 0
      ;;
  esac
done

if [ -z "$first_checkpoint" ] && [ ${#model_paths[@]} -eq 0 ]; then
  echo "play: --checkpoint or --models is required" >&2
  usage
  exit 2
fi

if [ "$has_spec" -eq 0 ]; then
  candidates=("${model_paths[@]:-}")
  if [ -n "$first_checkpoint" ]; then candidates=("$first_checkpoint" "${candidates[@]}"); fi
  # The model's recorded identity wins over the runs/<run>/ directory guess: an artifacts/<name>/
  # tree can share its name with a config directory without being that recipe. A checkpoint stores
  # the spec_sha256 its loader enforces; an ONNX export names parent_spec plus that file's SHA.
  # The directory guess only applies when the model records nothing.
  derivation=$("$PY" - "$ROOT" "${candidates[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1])

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def recorded(path):
    if path.suffix == '.onnx':
        manifest = path.with_name('manifest.json')
        if not manifest.exists():
            return None
        data = json.loads(manifest.read_text(encoding='utf-8'))
        parent = data.get('parent_spec')
        if not parent:
            # actor_onnx_export_v1 names the source checkpoint; its directory or recorded SHA
            # still identifies the recipe, so fall through to that path instead of failing.
            source = (data.get('checkpoint') or {}).get('path')
            if not source:
                return None
            source = Path(source)
            return recorded(source) or by_directory(source) if source.exists() else None
        parent = Path(parent)
        if not parent.exists():
            raise SystemExit(f'play: recorded ONNX parent spec is missing: {parent}')
        if data.get('spec_sha256') != sha(parent):
            raise SystemExit(f'play: ONNX manifest spec SHA does not match its parent_spec: {parent}')
        return parent
    if path.suffix not in ('.pt', '.pth'):
        return None
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=False)
    wanted = saved.get('spec_sha256') if isinstance(saved, dict) else None
    if not wanted:
        return None
    matches = [p for p in sorted((root / 'configs/experiments').glob('*/spec.json')) if sha(p) == wanted]
    if len(matches) > 1:
        raise SystemExit(f'play: {len(matches)} specs share spec_sha256 {wanted}; pass --spec')
    return matches[0] if matches else None

def by_directory(path):
    guess = root / 'configs/experiments' / path.parent.parent.name / 'spec.json'
    return guess if guess.is_file() else None

for raw in sys.argv[2:]:
    if not raw:
        continue
    path = Path(raw)
    if not path.exists():
        continue
    own, guess = recorded(path), by_directory(path)
    if own is not None:
        if guess is not None and guess.resolve() != own.resolve():
            print(f'play: ignoring {guess} from the directory name; the model records {own}', file=sys.stderr)
        print('recorded\t' + str(own.resolve()))
        break
    if guess is not None:
        print('directory\t' + str(guess.resolve()))
        break
PY
  ) || exit 2
  spec_source=${derivation%%$'\t'*}
  spec_path=${derivation#*$'\t'}
  case "$spec_source" in
    recorded) echo "play: --spec from the model's recorded spec identity -> ${spec_path#"$ROOT"/}" >&2 ;;
    directory) echo "play: --spec from the model directory name -> ${spec_path#"$ROOT"/}" >&2 ;;
    *)
      echo "play: no --spec given and the model records no parent spec; pass --spec explicitly" >&2
      exit 2
      ;;
  esac
  args=(--spec "$spec_path" "${args[@]}")
fi

if [ ! -f "$spec_path" ]; then
  echo "play: spec not found: $spec_path" >&2
  exit 2
fi

if [ "$has_armature" -eq 0 ]; then
  armature=$("$PY" - "$spec_path" <<'PY'
import json, sys
spec = json.loads(open(sys.argv[1], encoding='utf-8').read())
# An MCU spec plays its own execution chain and the training-fixed MCU leg armature;
# only legacy specs still take the armature level from motion_timing.configuration.
if (spec.get('mcu_execution') or {}).get('profile'):
    from motion_mcu_playback import MCU_LEG_ARMATURE_KG_M2
    print(MCU_LEG_ARMATURE_KG_M2)
else:
    declared = (spec.get('motion_timing') or {}).get('configuration') or {}
    print(declared.get('leg_armature_kg_m2', ''))
PY
)
  if [ -n "$armature" ]; then
  args+=(--leg-armature-m "$armature")
    echo "play: --leg-armature-m ${armature} from the spec-declared execution chain" >&2
  fi
fi

# Default playback plant for Motion31 MCU: the delivered full-v4 identified leg plant
# (named-body mass/inertia scale, motor-only clamp, per-side modal transport/response and a
# separate NET load). PLAY_PLANT_PROFILE=spec keeps the spec plant and re-enables the lighter
# bilateral response options below.
plant_profile=${PLAY_PLANT_PROFILE:-motion_v4_identified_v1}
leg_response=${PLAY_LEG_RESPONSE:-bilateral_modal_v1}
if [ "$has_leg_response" -eq 0 ] && [ "${has_plant_profile:-0}" -eq 0 ]; then
  motion31=$("$PY" - "$spec_path" <<'PY'
import json, sys
spec = json.loads(open(sys.argv[1], encoding='utf-8').read())
motion31 = (spec.get('interface_version') == 'multiskill_mcu_v2'
            and (spec.get('student_training') or {}).get('version') == 'student_motion31_training_v1'
            and bool(spec.get('mcu_execution')))
print(int(motion31))
PY
)
  if [ "$motion31" = "1" ] && [ "$plant_profile" = "motion_v4_identified_v1" ]; then
    args+=(--plant-profile motion_v4_identified_v1)
    echo "play: --plant-profile motion_v4_identified_v1 (full-v4 identified leg plant; PLAY_PLANT_PROFILE=spec keeps the spec plant)" >&2
  elif [ "$motion31" = "1" ] && [ "$leg_response" != "spec" ]; then
    args+=(--leg-response "$leg_response")
    echo "play: --leg-response ${leg_response} (delivered bilateral modal leg response; PLAY_LEG_RESPONSE=spec restores the spec chain)" >&2
    leg_physics=${PLAY_LEG_PHYSICS:-bilateral_modal_v1}
    if [ "$leg_physics" != "spec" ]; then
      args+=(--leg-physics "$leg_physics")
      echo "play: --leg-physics ${leg_physics} (leg joint damping replaced by the delivered modal B; PLAY_LEG_PHYSICS=spec keeps the spec plant)" >&2
    fi
  elif [ "$motion31" != "1" ]; then
    echo "play: --plant-profile/--leg-response skipped; the identified handoff is Motion31 MCU only" >&2
  fi
fi

if [ "${PLAY_HEADLESS:-0}" != "1" ]; then
  if [ "$has_viewer" -eq 0 ]; then
    args=(--viewer "${args[@]}")
    echo "play: --viewer (window) added by default" >&2
  fi
  # Browser control is the default; keyboard/scored playback/headless opt out.
  if [ "$has_panel" -eq 0 ] && [ "$has_keyboard" -eq 0 ] && [ "$has_commands" -eq 0 ] && [ "${PLAY_PANEL:-1}" != "0" ]; then
    args+=(--panel --panel-host "$PANEL_HOST" --panel-port "$PANEL_PORT")
    echo "play: --panel on http://${PANEL_HOST}:${PANEL_PORT}/ added by default" >&2
    if [ "$has_max_speed" -eq 0 ]; then args+=(--max-speed .5); fi
    if [ "$has_max_yaw" -eq 0 ]; then args+=(--max-yaw .5); fi
  fi
fi

cmd=("$PY" scripts/validate_native_mujoco.py --run "${args[@]}")
echo "play: ${cmd[*]}" >&2
if [ "${PLAY_DRY_RUN:-0}" = "1" ]; then exit 0; fi
exec "${cmd[@]}"
