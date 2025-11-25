# CUDA версия решателя дифракции

Эта директория содержит перенос вычислительно самых тяжёлых частей CPU-проекта (`cpp_version`) на CUDA. Построение матрицы интегрального уравнения и вектора правой части выполняется на GPU, а решение СЛАУ по‑прежнему происходит на CPU (метод Гаусса), что позволяет честно сравнивать время построения матрицы.

## Структура

- `include/` — общие заголовки с CUDA-макросами (`CUDA_HOST_DEVICE`), доступные как на CPU, так и на GPU.
- `src/` — исходники:
  - `main.cu` — CLI-утилита, повторяющая сценарий `cpp_version/src/main.cpp`.
  - `DifrOnLentaCuda.cu` — обёртка, управляющая памятью GPU и вызывающая CUDA-ядра.
  - `Bessel.cu`, `Chebyshev.cu` — функции Бесселя и полиномы Чебышева с разметкой `__host__ __device__`.
  - `Matrix.cpp`, `Gauss.cpp` — CPU-компоненты (решение СЛАУ).
- `colab/build_cuda.sh` — сценарий сборки/запуска под Google Colab.

## Требования

- CUDA Toolkit 11.0+ (для Google Colab достаточно включить GPU в настройках рантайма).
- CMake ≥ 3.23 и любой поддерживаемый генератор (Ninja или Makefiles).
- Компилятор C++17.

## Сборка и запуск (локально)

```bash
cd cuda_version
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/CudaDiffr
```

По умолчанию используется архитектура `sm_75`. Можно переопределить её посредством `-DCMAKE_CUDA_ARCHITECTURES=86` и т.п.

## Google Colab

### Только CUDA-версия

```bash
!bash cuda_version/colab/build_cuda.sh
```

Сценарий подтянет CMake/компиляторы, соберёт `CudaDiffr` и выведет CSV для построения графиков.

### Сравнение CUDA vs C++

```bash
!python cuda_version/colab/run_benchmark.py
```

Python-скрипт:

- устанавливает `matplotlib`, `pandas` (если их нет);
- собирает **обе** версии через CMake (`cpp_version/build_colab`, `cuda_version/build`);
- запускает их и парсит CSV-блоки из stdout;
- сохраняет JSON/CSV с результатами и строит три графика (`plots/matrix_compare.png`, `plots/total_compare.png`, `plots/speedup.png`).

Все артефакты находятся в `cuda_version/plots`.

## Сравнение с CPU версией

- В Windows остаются актуальны `.bat`-скрипты внутри `cpp_version`.
- Для автоматического сравнения на Linux/Colab используйте `colab/run_benchmark.py`.
- Обе реализации печатают CSV для набора `N = {10,20,30,40,50}`, поэтому графики накладываются корректно.


