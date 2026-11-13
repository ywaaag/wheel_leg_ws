"""Same firmware C bodies executing as CUDA rows, with explicit host snapshots."""

import ctypes as C

import numpy as np
import torch

from native_stand.simulation.jump_cuda_build import build_cuda
from native_stand.simulation.jump_device_c import abi_field
from native_stand.simulation.jump_shared_c import Config, Frame, Input, State


class CudaControl:
    def __init__(self, host, output, device):
        self.host, self.device, self.n = host, device, host.n
        if torch.cuda.get_device_capability(device) != (8, 9):
            raise ValueError("CUDA firmware build currently validated for sm89 only")
        self.lib, self.build_receipt = build_cuda(output)
        self.lib.launch.argtypes = [C.c_int] + [C.c_void_p] * 8 + [C.c_uint, C.c_uint, C.c_void_p]
        self.lib.launch.restype = C.c_int
        for k, size in enumerate(
            (4, C.sizeof(Config), C.sizeof(Input), C.sizeof(State), C.sizeof(Frame))
        ):
            if self.lib.abi_size(k) != size:
                raise ValueError("CUDA ABI layout mismatch")
        self.states = self.words(host.states)
        self.inputs = self.words(host.inputs)
        self.source_inputs = self.inputs.clone()
        self.config = self.words(host.config)
        self.frames = self.words(host.frames)
        self.snapshot = self.states.clone()
        self.actor = torch.zeros(self.n, 37, device=device)
        self.mask = torch.zeros(self.n, dtype=torch.int32, device=device)
        self.rc = torch.zeros_like(self.mask)

    def words(self, value):
        return (
            torch.from_numpy(np.frombuffer(bytes(value), np.int32).copy())
            .to(self.device)
            .reshape(
                -1, C.sizeof(type(value)._type_ if isinstance(value, C.Array) else type(value)) // 4
            )
        )

    def field(self, name, *, inputs=False):
        return abi_field(self.inputs if inputs else self.states, Input if inputs else State, name)

    def set_input(self, name, value, *, source=False):
        words = self.source_inputs if source else self.inputs
        self.write(words, Input, name, value)

    @staticmethod
    def write(words, kind, name, value):
        typ = dict(kind._fields_)[name]
        count = C.sizeof(typ) // 4
        offset = getattr(kind, name).offset // 4
        if typ is C.c_float or getattr(typ, "_type_", None) is C.c_float:
            data = (
                torch.as_tensor(value, device=words.device, dtype=torch.float32)
                .contiguous()
                .view(torch.int32)
            )
        else:
            data = torch.as_tensor(value, device=words.device, dtype=torch.int64).to(torch.int32)
        words[:, offset : offset + count] = data.reshape(-1, count)

    def launch(self, op, tick=0, action=None, mask=None):
        args = (
            self.states,
            self.inputs,
            self.config,
            self.frames,
            self.snapshot,
            self.actor if action is None else action,
            self.mask if mask is None else mask,
            self.rc,
        )
        code = self.lib.launch(
            op,
            *[x.data_ptr() for x in args],
            self.n,
            tick & 0xFFFFFFFF,
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        if code:
            raise RuntimeError(f"CUDA firmware launch failed {code}")

    def sync_to_host(self):
        for target, source in (
            (self.host.states, self.states),
            (self.host.inputs, self.inputs),
            (self.host.frames, self.frames),
        ):
            raw = source.cpu().numpy().tobytes()
            C.memmove(C.addressof(target), raw, len(raw))

    def reset_host_rows(self, ids):
        index = torch.as_tensor(ids, device=self.device)
        self.states[index] = self.words(self.host.states)[index]
        self.inputs[index] = self.words(self.host.inputs)[index]
        self.source_inputs[index] = self.inputs[index]
