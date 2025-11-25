## Project Navigation

### Top-Level
- `cpp_version/` – чистая C++ реализация решателя; включает исходники, заголовки, батники и анализ времени.
- `cuda_version/` – гибрид CUDA/C++ версия с поддержкой GPU, colab-скриптами и Python-утилитами для бенчмарков.
- `Diffraction-main/` – WinForms-приложение (.NET) с интерфейсом на C#.
- `local_benchmark.ipynb` / `diff-cuda-c++.ipynb` – ноутбуки для экспериментов и сравнений.

### `cpp_version/`
- `src/` – основные реализации (`Matrix.cpp`, `Gauss.cpp`, `DifrOnLenta.cpp`, `main.cpp`).
- `include/` – заголовочные файлы с описаниями матриц, многочленов, Гаусса и т.д.
- `build/` – артефакты после сборки (`diffraction_solver.exe`, `output.txt`).
- `build.bat`, `run.bat`, `compile_and_run.bat`, `run_and_plot.bat`, `START_HERE.bat` – сценарии для Windows.
- `plot_timing.py`, `timing_data.csv`, `timing_analysis.png`, `complexity_analysis.png` – инструменты и результаты профилирования.

### `cuda_version/`
- `src/` – CUDA-ядра и CPU-поддержка (`main.cu`, `DifrOnLentaCuda.cu`, `Bessel.cu` и др.).
- `include/` – заголовки для CUDA/CPU частей (`CudaCommon.h`, `DifrOnLentaCuda.h`, и т.д.).
- `colab/` – скрипты `build_cuda.sh`, `run_benchmark.py` для автоматических бенчмарков в Colab.
- `plots/` (создаётся после запуска бенчмарков) – CSV/PNG результаты сравнений CPU vs CUDA.

### `Diffraction-main/` (WinForms)
- `Form1.cs`, `Form2.cs` + `*.Designer.cs` – формы GUI.
- `Program.cs` – точка входа WinForms.
- `bin/`, `obj/` – стандартные .NET артефакты (не редактируются вручную).
- `ЗАПУСТИТЬ.bat`, `ОЧИСТИТЬ_И_СОБРАТЬ.bat` – сценарии для сборки/запуска GUI.

### Сценарии использования
1. **Быстрые тесты CPU** – зайдите в `cpp_version/`, используйте `START_HERE.bat` или ноутбук для компиляции/запуска.
2. **GPU бенчмарки** – `cuda_version/colab/run_benchmark.py` или `diff-cuda-c++.ipynb` с CUDA сборкой.
3. **Графический интерфейс** – проект `Diffraction-main/` в Visual Studio для работы с WinForms.
4. **Ноутбук** – `diff-cuda-c++.ipynb` теперь позволяет выбирать коэффициенты, пересобирать CPU/CUDA и видеть встроенные фрагменты кода.

