"""Local browser panel for manual Motion preview: stdlib only, loopback only.

The panel never touches Isaac/PhysX objects. The GUI loop owns the policy and
exchanges two things with this module through Plan:

* commands (forward speed, yaw rate, height target, 31D skill/jump one-hot, stop,
  center, reset) written by the browser and consumed by the GUI loop once per step;
* telemetry published by the GUI loop and served read-only to the browser.

Explicit opt-in only (``--panel`` or ``NATIVE_MOTION_PANEL=1``); default port 8765
(``NATIVE_MOTION_PANEL_PORT``). Caps declared on the command line stay authoritative:
the Isaac browser may only lower them. MuJoCo manual preview allows finite
positive session caps to be edited above the initial values.
"""
import json
import errno
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VERSION = 'native_motion_panel_v3'
DEFAULT_PORT = 8765
ENABLED_ENV = 'NATIVE_MOTION_PANEL'
PORT_ENV = 'NATIVE_MOTION_PANEL_PORT'
MAX_REQUEST_BYTES = 4096
# The last six Actor dimensions are the skill/jump one-hot; both stay editable so the
# 31D input can be inspected, but the trained task is Motion/NONE.
SKILLS = ('Motion', 'Recovery', 'Jump')
JUMPS = ('NONE', 'SMALL', 'BIG')
HEIGHT_BOUNDS = (.20, .45)
COMMAND_KEYS = ('vx', 'yaw', 'height', 'skill', 'jump', 'model', 'stop', 'center', 'reset',
                'max_speed', 'max_yaw_speed', 'paused')
FIXED_MOVEMENT_KEYS = ('vx', 'yaw', 'max_speed', 'max_yaw_speed', 'height', 'skill', 'jump')
HTML_PATH = Path(__file__).with_name('motion_panel.html')


def enabled(flag):
    """Explicit opt-in; the environment variable is only a convenience alias."""
    if flag:
        return True
    return os.environ.get(ENABLED_ENV, '0').strip().lower() in ('1', 'true', 'yes')


def port_number(explicit=None):
    raw = explicit if explicit is not None else os.environ.get(PORT_ENV, DEFAULT_PORT)
    try:
        number = int(raw)
    except (TypeError, ValueError):
        raise ValueError('panel port must be an integer')
    if not 1 <= number <= 65535:
        raise ValueError('panel port must be within 1..65535')
    return number


def _finite(name, value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError('%s must be numeric' % name)
    if number != number or number in (float('inf'), float('-inf')):
        raise ValueError('%s must be finite' % name)
    return number


def _bounded(name, value, low, high):
    number = _finite(name, value)
    if not low <= number <= high:
        raise ValueError('%s must be within %s..%s' % (name, low, high))
    return number


def _clamped(name, value, low, high):
    """Browser targets are clamped into the session caps; caps themselves stay strict."""
    return max(low, min(high, _finite(name, value)))


def json_default(value):
    """Telemetry producers mix numpy/torch values with plain floats; encode them as JSON natives."""
    for name in ('tolist', 'item'):
        convert = getattr(value, name, None)
        if callable(convert):
            try:
                return convert()
            except (TypeError, ValueError):
                break
    raise TypeError('Object of type %s is not JSON serializable' % type(value).__name__)


class Plan:
    """Locked holder shared by the GUI loop (owner) and the HTTP thread."""

    def __init__(self, max_speed, max_yaw_speed, *, mode='Motion', backend='Isaac', models=None, model=None):
        if mode not in ('Motion','Recovery'):raise ValueError('unknown panel mode')
        if backend not in ('Isaac','MuJoCo'):raise ValueError('unknown panel backend')
        self.backend=backend
        self.mode=mode
        # Model switching compares checkpoints in the same plant; names are explicit, never inferred.
        self.models=list(models) if models else ['main']
        if len(set(self.models))!=len(self.models) or not self.models:raise ValueError('panel model names must be unique')
        if model is None:model=self.models[0]
        if model not in self.models:raise ValueError('unknown panel model %r' % model)
        self._model=model
        self._lock = threading.Lock()
        self.editable_limits = backend=='MuJoCo' and mode=='Motion'
        initial_ceiling = float('inf') if self.editable_limits else 4.
        self._declared = {'max_speed': _bounded('max_speed', max_speed, 1e-3, initial_ceiling),
                          'max_yaw_speed': _bounded('max_yaw_speed', max_yaw_speed, 0., initial_ceiling)}
        self._limits = dict(self._declared)
        self._target = [0., 0.]
        self._height = .34
        self._skill = 'Motion'
        self._jump = 'NONE'
        self._pending = {'stop': False, 'center': False, 'reset': False}
        self._telemetry = {}
        self.requests = 0
        # Pause is opt-in: only a backend that can actually freeze its plant sets this.
        self.pause_supported = False
        self._paused = False

    def apply(self, payload):
        """Validate and apply one browser request; returns the applied subset."""
        if not isinstance(payload, dict):
            raise ValueError('command payload must be an object')
        unknown = [key for key in payload if key not in COMMAND_KEYS]
        if unknown:
            raise ValueError('unknown command fields: %s' % ','.join(sorted(unknown)))
        if self.mode=='Recovery' and any(k in payload for k in FIXED_MOVEMENT_KEYS):
            raise ValueError('Recovery uses fixed zero commands; movement controls are disabled')
        applied = {}
        with self._lock:
            if 'max_speed' in payload:
                value = _bounded('max_speed', payload['max_speed'], 1e-3, float('inf') if self.editable_limits else self._declared['max_speed'])
                self._limits['max_speed'] = value
                applied['max_speed'] = value
                self._target[0] = max(-value, min(value, self._target[0]))
            if 'max_yaw_speed' in payload:
                value = _bounded('max_yaw_speed', payload['max_yaw_speed'], 0., float('inf') if self.editable_limits else self._declared['max_yaw_speed'])
                self._limits['max_yaw_speed'] = value
                applied['max_yaw_speed'] = value
                self._target[1] = max(-value, min(value, self._target[1]))
            if 'vx' in payload:
                value = _clamped('vx', payload['vx'], -self._limits['max_speed'], self._limits['max_speed'])
                self._target[0] = value
                applied['vx'] = value
            if 'yaw' in payload:
                value = _clamped('yaw', payload['yaw'], -self._limits['max_yaw_speed'], self._limits['max_yaw_speed'])
                self._target[1] = value
                applied['yaw'] = value
            if 'height' in payload:
                value = _bounded('height', payload['height'], *HEIGHT_BOUNDS)
                self._height = value
                applied['height'] = value
            if 'skill' in payload:
                if payload['skill'] not in SKILLS:
                    raise ValueError('skill must be one of %s' % ','.join(SKILLS))
                self._skill = payload['skill']
                applied['skill'] = self._skill
            if 'jump' in payload:
                if payload['jump'] not in JUMPS:
                    raise ValueError('jump must be one of %s' % ','.join(JUMPS))
                self._jump = payload['jump']
                applied['jump'] = self._jump
            if 'model' in payload:
                if payload['model'] not in self.models:
                    raise ValueError('model must be one of %s' % ','.join(self.models))
                self._model = payload['model']
                applied['model'] = self._model
            if payload.get('stop'):
                self._target = [0., 0.]
                self._pending['stop'] = True
                applied['stop'] = True
            if payload.get('center'):
                self._target = [0., 0.]
                self._pending['center'] = True
                applied['center'] = True
            if payload.get('reset'):
                self._pending['reset'] = True
                applied['reset'] = True
            if 'paused' in payload:
                if type(payload['paused']) is not bool:
                    raise ValueError('paused must be a boolean')
                if not self.pause_supported:
                    raise ValueError('this backend has no pause control')
                self._paused = payload['paused']
                applied['paused'] = self._paused
            self.requests += 1
        return applied

    @property
    def paused(self):
        return self._paused

    def take(self):
        """Snapshot for the GUI loop; one-shot flags are consumed exactly once."""
        with self._lock:
            row = {'vx': self._target[0], 'yaw': self._target[1],
                   'height': self._height, 'skill': self._skill, 'jump': self._jump, 'model': self._model,
                   'max_speed': self._limits['max_speed'], 'max_yaw_speed': self._limits['max_yaw_speed'],
                   'stop': self._pending['stop'], 'center': self._pending['center'], 'reset': self._pending['reset']}
            self._pending = {'stop': False, 'center': False, 'reset': False}
        return row

    def publish(self, telemetry):
        with self._lock:
            self._telemetry = dict(telemetry)

    def snapshot(self):
        with self._lock:
            return {'version': VERSION, 'mode':self.mode, 'backend':self.backend, 'editable_limits':self.editable_limits, 'target': {'vx': self._target[0], 'yaw': self._target[1]},
                    'height': self._height, 'skill': self._skill, 'jump': self._jump,
                    'model': self._model, 'models': list(self.models),
                    'skills': list(SKILLS), 'jumps': list(JUMPS), 'height_bounds': list(HEIGHT_BOUNDS),
                    'limits': dict(self._limits), 'declared_limits': dict(self._declared),
                    'requests': self.requests, 'paused': self._paused,
                    'pause_supported': self.pause_supported, 'telemetry': dict(self._telemetry)}


def handler_for(plan, html):
    """HTTP handler bound to one Plan; all responses are JSON except the page."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        server_version = VERSION

        def log_message(self, *args):
            pass

        def _send(self, code, body, content_type):
            data = body if isinstance(body, bytes) else body.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code, payload):
            self._send(code, json.dumps(payload, default=json_default), 'application/json')

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self._send(200, html, 'text/html; charset=utf-8')
            elif self.path == '/telemetry':
                self._json(200, plan.snapshot())
            elif self.path == '/health':
                self._json(200, {'status': 'ok', 'version': VERSION})
            else:
                self._json(404, {'error': 'unknown path'})

        def do_POST(self):
            if self.path != '/command':
                self._json(404, {'error': 'unknown path'})
                return
            try:
                length = int(self.headers.get('Content-Length') or 0)
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_REQUEST_BYTES:
                self._json(413, {'error': 'unsupported request size'})
                return
            try:
                payload = json.loads(self.rfile.read(length) or b'{}')
            except ValueError:
                self._json(400, {'error': 'invalid JSON'})
                return
            try:
                applied = plan.apply(payload)
            except ValueError as error:
                self._json(400, {'error': str(error)})
                return
            self._json(200, {'applied': applied, **plan.snapshot()})

    return Handler


def start(plan, port, host='127.0.0.1'):
    """Serve on the loopback interface by default; returns (server, thread).

    A non-loopback ``host`` must be an address the caller deliberately owns: the
    page accepts command POSTs, so exposing it on a LAN interface widens who can
    drive the preview.
    """
    html = HTML_PATH.read_text()
    try:
        server = ThreadingHTTPServer((host, port), handler_for(plan, html))
    except OSError as exc:
        if exc.errno in (errno.EADDRINUSE, errno.EACCES):
            raise RuntimeError(
                'panel address %s:%s is not available (%s); another preview is probably still '
                'running - stop it or pass a different --panel-port' % (host, port, exc.strerror)) from exc
        raise
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name='motion-panel', daemon=True)
    thread.start()
    return server, thread


def stop(server):
    if server is None:
        return
    server.shutdown()
    server.server_close()
