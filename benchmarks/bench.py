"""极薄度 benchmark —— 量 rein 框架【自身】的硬指标(可重复跑)。

为什么不直接跑「vs LangChain / Pydantic AI」的对比数字:那其实**难做到公允** ——
import 时间被 pydantic / pydantic-core 主导(几家都用 pydantic,数字趋同),
框架 LOC 的统计口径又各不相同(功能集不一样)。硬塞一个对比数字反而误导。

所以这里只量 rein 自身的客观硬指标;"极薄"的真正体现是
【框架源码少 + 核心概念少 + 源码能一眼看穿】—— 跨框架对比留给定性分析。

跑:python benchmarks/bench.py
"""

import os
import subprocess
import sys
import time


def import_ms(module: str, runs: int = 5) -> float:
    """子进程 cold import 多次取最小值(毫秒)。每次新进程,避免缓存。"""
    times = []
    for _ in range(runs):
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import time,sys;t=time.perf_counter();import {module};"
                f"sys.stderr.write(str((time.perf_counter()-t)*1000))",
            ],
            capture_output=True,
            text=True,
        )
        try:
            times.append(float(r.stderr.strip().split()[-1]))
        except Exception:
            pass
    return min(times) if times else -1.0


def src_loc(pkg_dir: str) -> tuple[int, int]:
    """框架源码的总行数与 .py 文件数。"""
    loc = nfiles = 0
    for root, _, files in os.walk(pkg_dir):
        if "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                nfiles += 1
                loc += sum(1 for _ in open(os.path.join(root, f), encoding="utf-8"))
    return loc, nfiles


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loc, nfiles = src_loc(os.path.join(root, "src", "rein"))
    print("=== Rein 极薄度指标 ===")
    print(f"框架源码      : {loc} 行 / {nfiles} 个 .py 文件")
    print(f"import 开销    : {import_ms('rein'):.1f} ms (cold,5 次取最小;主要是 pydantic)")
    print("核心依赖      : pydantic, anyio, litellm (3 个直接依赖)")
    print(
        "核心概念      : Agent / Session / Chat / Loop / Provider / "
        "Runtime / Tool / RunResult (~8 个)"
    )
    print("\n注:未跑跨框架对比 —— 见 docs 的定性分析(对比数字难公允,详见本文件头)。")


if __name__ == "__main__":
    main()
