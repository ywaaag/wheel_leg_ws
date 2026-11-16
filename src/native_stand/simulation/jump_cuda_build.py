"""Compile the SHA-verified scalar firmware core for one CUDA thread per row.

Only CUDA execution qualifiers are generated in a private build directory;
firmware sources remain unchanged. No fast math or fused multiply-add.
"""

import ctypes as C
import re
import shutil
import subprocess
from pathlib import Path

from native_stand.artifacts.io import file_sha256, write_json
from native_stand.simulation.jump_build import SOURCE, build

WRAPPER = r"""
#include <cuda_runtime.h>
#include "rl_motion31.c"
#include "rl_control.c"
#include "rl_jump_core.c"
__global__ void run(int op, RLJumpState *s, RLJumpInput *i, const RLJumpConfig *cfg,
 RLJumpFrame *f, RLJumpState *old, float *a, unsigned *mask, int *rc, unsigned n, unsigned now) {
 unsigned k=blockIdx.x*blockDim.x+threadIdx.x;
 if(k>=n) return;
 if(op==0) {RLJumpInit(s+k,now);return;}
 if(op==1) {rc[k]=RLJumpStep(s+k,i+k,cfg);return;}
 if(op==2) {memset(a+37*k,0,37*sizeof(float));mask[k]=RLJumpEncode(s+k,i+k,a+37*k)==0;old[k]=s[k];return;}
 if(op==3) {rc[k]=mask[k]?RLJumpPublishMean(s+k,i+k,old+k,old[k].adopted_sequence+1,a+6*k):1;return;}
 if(op==4) {
   rc[k]=RLJumpPd(s+k,i+k,f+k);
   if(rc[k]<0) memset(f+k,0,sizeof(RLJumpFrame));
   if(rc[k]==1) RLJumpUpstreamMotorCommand(i+k,i[k].upstream_target,f+k);
 }
}
extern "C" int launch(int op,void*s,void*i,void*c,void*f,void*old,void*a,void*mask,void*rc,unsigned n,unsigned now,void*stream) {
 run<<<(n+127)/128,128,0,(cudaStream_t)stream>>>(op,(RLJumpState*)s,(RLJumpInput*)i,(RLJumpConfig*)c,(RLJumpFrame*)f,(RLJumpState*)old,(float*)a,(unsigned*)mask,(int*)rc,n,now);
 return (int)cudaGetLastError();
}
extern "C" unsigned abi_size(unsigned k) {
 switch(k) {case 0:return RL_JUMP_ABI;case 1:return sizeof(RLJumpConfig);case 2:return sizeof(RLJumpInput);case 3:return sizeof(RLJumpState);case 4:return sizeof(RLJumpFrame);default:return 0;}
}
"""


def build_cuda(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reference = build(output / "reference.so")
    pattern = re.compile(
        r"^(static\s+)?(void|int|float|double|uint32_t|RLControlDecision)\s+([A-Za-z_]\w*)\s*\(",
        re.M,
    )
    for name in (
        "rl_shadow.h",
        "rl_control.h",
        "rl_motion31.h",
        "rl_jump_core.h",
        "rl_control.c",
        "rl_motion31.c",
        "rl_jump_core.c",
    ):
        text = (SOURCE / name).read_text()
        text = pattern.sub(lambda m: (m[1] or "") + "__device__ " + m[2] + " " + m[3] + "(", text)
        (output / name).write_text(text)
    (output / "wrapper.cu").write_text(WRAPPER)
    compiler = shutil.which("nvcc")
    if compiler is None:
        raise RuntimeError("CUDA compiler unavailable")
    args = [
        compiler,
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "--fmad=false",
        "--prec-div=true",
        "--prec-sqrt=true",
        "-O2",
        "-arch=sm_89",
        "-DRL_JUMP_DIRECT_V1=1",
        "-DRL_JUMP_NOMINAL_V1=0",
        str(output / "wrapper.cu"),
        "-o",
        str(output / "control.so"),
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    (output / "compiler.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("CUDA control compile failed: " + result.stderr[-3000:])
    receipt = dict(
        reference=reference,
        compiler=compiler,
        compiler_sha256=file_sha256(compiler),
        flags=args,
        library_sha256=file_sha256(output / "control.so"),
        generated={
            p.name: file_sha256(p) for p in output.iterdir() if p.suffix in {".h", ".c", ".cu"}
        },
        source_semantics="only device qualifier insertion; no body rewrite",
    )
    write_json(output / "receipt.json", receipt)
    lib = C.CDLL(str(output / "control.so"))
    return lib, receipt
