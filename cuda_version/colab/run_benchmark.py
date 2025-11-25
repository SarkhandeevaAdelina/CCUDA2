#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab helper script that
1) Installs build dependencies (cmake, build-essential, ninja, python libs)
2) Builds CPU (cpp_version) and CUDA (cuda_version) solvers via CMake
3) Runs executables, captures timing CSV blocks
4) Plots CPU vs CUDA timings and speedups

Usage inside Google Colab:
    !python cuda_version/colab/run_benchmark.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CPU_SRC = ROOT.parent / "cpp_version"
CUDA_SRC = ROOT
CPU_BUILD = CPU_SRC / "build_colab"
CUDA_BUILD = CUDA_SRC / "build"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: List[str], cwd: Path | None = None, env: Dict[str, str] | None = None) -> str:
    print(f"\n[cmd] {' '.join(cmd)}")
    process = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        text=True,
    )
    print(process.stdout)
    return process.stdout


def ensure_packages() -> None:
    print("Проверка наличия CMake/compilers...")
    try:
        run_cmd(["cmake", "--version"])
    except (OSError, subprocess.CalledProcessError):
        run_cmd(["sudo", "apt-get", "update"])
        run_cmd(
            ["sudo", "apt-get", "install", "-y", "cmake", "build-essential", "ninja-build"]
        )

    print("Установка python-библиотек для графиков...")
    run_cmd([sys.executable, "-m", "pip", "install", "--quiet", "matplotlib", "pandas"])


def ensure_gpu() -> None:
    try:
        run_cmd(["nvidia-smi"])
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            "GPU недоступен. В Colab включите GPU (Runtime → Change runtime type → GPU)."
        ) from exc


def configure_and_build(src_dir: Path, build_dir: Path, cmake_args: List[str]) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        ["cmake", "-S", str(src_dir), "-B", str(build_dir), "-G", "Ninja", *cmake_args]
    )
    run_cmd(["cmake", "--build", str(build_dir), "--config", "Release"])


def parse_csv_block(text: str) -> List[Tuple[int, float, float, float]]:
    csv_pattern = re.compile(r"^\s*N,\w+", re.MULTILINE)
    header_match = csv_pattern.search(text)
    if not header_match:
        raise RuntimeError("Не найден CSV-блок в выводе программы")

    start = header_match.start()
    lines = text[start:].strip().splitlines()
    header = lines[0].strip().split(",")
    rows: List[Tuple[int, float, float, float]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != len(header):
            continue
        try:
            n = int(parts[0])
            values = tuple(float(p) for p in parts[1:])
            rows.append((n, *values))
        except ValueError:
            continue
    return rows


@dataclass
class TimingDataset:
    label: str
    rows: List[Tuple[int, float, float, float]]
    header: Tuple[str, str, str, str]

    def to_dict(self) -> Dict[str, List[float]]:
        data: Dict[str, List[float]] = {self.header[0]: []}
        for h in self.header[1:]:
            data[h] = []
        for row in self.rows:
            data[self.header[0]].append(row[0])
            for idx, h in enumerate(self.header[1:], start=1):
                data[h].append(row[idx])
        return data


def run_solver(executable: Path) -> str:
    return run_cmd([str(executable)])


def dataset_from_output(text: str, label: str, header: Tuple[str, str, str, str]) -> TimingDataset:
    rows = parse_csv_block(text)
    return TimingDataset(label=label, rows=rows, header=header)


def plot_comparison(cpu: TimingDataset, cuda: TimingDataset) -> None:
    import pandas as pd

    cpu_df = pd.DataFrame(cpu.rows, columns=cpu.header).rename(
        columns={
            cpu.header[1]: "MatrixTime_ms_CPU",
            cpu.header[2]: "SolveTime_ms_CPU",
            cpu.header[3]: "TotalTime_ms_CPU",
        }
    )
    cuda_df = pd.DataFrame(cuda.rows, columns=cuda.header).rename(
        columns={
            cuda.header[1]: "MatrixGPU_ms_CUDA",
            cuda.header[2]: "SolveTime_ms_CUDA",
            cuda.header[3]: "Total_ms_CUDA",
        }
    )
    merged = cpu_df.merge(cuda_df, on="N")

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        merged["N"],
        merged["MatrixTime_ms_CPU"],
        "o-",
        label="Матрица CPU",
        linewidth=2,
    )
    ax.plot(
        merged["N"],
        merged["MatrixGPU_ms_CUDA"],
        "s-",
        label="Матрица GPU",
        linewidth=2,
    )
    ax.set_xlabel("N")
    ax.set_ylabel("Время (мс)")
    ax.set_title("Сравнение построения матрицы")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "matrix_compare.png", dpi=300)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(
        merged["N"],
        merged["TotalTime_ms_CPU"],
        "o-",
        label="Всего CPU",
        linewidth=2,
    )
    ax2.plot(
        merged["N"],
        merged["Total_ms_CUDA"],
        "s-",
        label="Всего CUDA",
        linewidth=2,
    )
    ax2.set_xlabel("N")
    ax2.set_ylabel("Время (мс)")
    ax2.set_title("Сравнение полного времени")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "total_compare.png", dpi=300)

    # Speedup plots
    merged["Speedup_matrix"] = merged["MatrixTime_ms_CPU"] / merged["MatrixGPU_ms_CUDA"]
    merged["Speedup_total"] = merged["TotalTime_ms_CPU"] / merged["Total_ms_CUDA"]
    
    # Use log scale or line plot if N varies widely
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    # Use line plot for speedup as N range is large now (10..500)
    ax3.plot(merged["N"], merged["Speedup_matrix"], 'o-', label="Ускорение матрицы", linewidth=2)
    ax3.plot(merged["N"], merged["Speedup_total"], 's-', label="Ускорение всего приложения", linewidth=2)
    
    ax3.set_xlabel("N")
    ax3.set_ylabel("Ускорение (раз)")
    ax3.set_title("Ускорение GPU относительно CPU (чем выше, тем лучше)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(PLOTS_DIR / "speedup.png", dpi=300)

    merged.to_csv(PLOTS_DIR / "combined_results.csv", index=False)
    print(f"Графики сохранены в {PLOTS_DIR}")


def main() -> None:
    ensure_packages()
    ensure_gpu()

    configure_and_build(CPU_SRC, CPU_BUILD, ["-DCMAKE_BUILD_TYPE=Release"])
    cpu_output = run_solver(CPU_BUILD / "diffraction_solver")
    cpu_dataset = dataset_from_output(
        cpu_output,
        "CPU",
        ("N", "MatrixTime_ms", "SolveTime_ms", "TotalTime_ms"),
    )

    configure_and_build(
        CUDA_SRC,
        CUDA_BUILD,
        ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75"],
    )
    cuda_output = run_solver(CUDA_BUILD / "CudaDiffr")
    # Note: CSV header from main.cu is now N,MatrixGPU_ms,SolveCPU_ms,Total_ms
    # But we know the 3rd column is now SolveGPU_ms implicitly, though header in main.cu might stay "SolveCPU_ms" unless I changed it.
    # I DID NOT change the CSV header in main.cu to avoid breaking the parser if I'm lazy, 
    # BUT I DID change it in the print statement in main.cu to "SolveCPU_ms" (Wait, I need to check what I wrote).
    # Checked main.cu: I wrote `std::cout << "N,MatrixGPU_ms,SolveCPU_ms,Total_ms\n";`
    # So the parser still sees SolveCPU_ms. 
    # In this script I map it to "SolveTime_ms_CUDA".
    cuda_dataset = dataset_from_output(
        cuda_output,
        "CUDA",
        ("N", "MatrixGPU_ms", "SolveCPU_ms", "Total_ms"),
    )

    with open(PLOTS_DIR / "cpu_results.json", "w", encoding="utf-8") as f:
        json.dump(cpu_dataset.to_dict(), f, ensure_ascii=False, indent=2)
    with open(PLOTS_DIR / "cuda_results.json", "w", encoding="utf-8") as f:
        json.dump(cuda_dataset.to_dict(), f, ensure_ascii=False, indent=2)

    plot_comparison(cpu_dataset, cuda_dataset)


if __name__ == "__main__":
    main()
