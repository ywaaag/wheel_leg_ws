"""python -m native_stand 的统一入口；与 wheel-leg 共用同一个parser。"""

from native_stand.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
