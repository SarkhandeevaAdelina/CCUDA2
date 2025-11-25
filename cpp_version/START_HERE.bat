@echo off
setlocal
chcp 65001 > nul
set "PATH=C:\msys64\mingw64\bin;%PATH%"

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build"
set "SRC_DIR=%SCRIPT_DIR%src\"
set "INCLUDE_DIR=%SCRIPT_DIR%include"

if not exist "%BUILD_DIR%" (
    echo Ошибка: папка build не найдена
    echo Ожидаемый путь: %BUILD_DIR%
    echo.
    pause
    exit /b 1
)

:menu
cls
echo.
echo Решатель дифракции на ленте
echo.
echo Выберите действие:
echo.
echo 1. Быстрый запуск
echo 2. Запуск с графиками
echo 3. Только компиляция
echo 4. Только запуск
echo 5. Выход
echo.
set /p choice="Ваш выбор (1-5): "

if "%choice%"=="1" goto quick_run
if "%choice%"=="2" goto run_with_plots
if "%choice%"=="3" goto compile_only
if "%choice%"=="4" goto run_only
if "%choice%"=="5" goto end
goto invalid

:quick_run
cls
echo.
echo Быстрый запуск
echo.
call "%SCRIPT_DIR%compile_and_run.bat"
goto end

:run_with_plots
cls
echo.
echo Запуск с построением графиков
echo.
call "%SCRIPT_DIR%run_and_plot.bat"
goto end

:compile_only
cls
echo.
echo Компиляция проекта
echo.
pushd "%BUILD_DIR%" > nul
g++ -std=c++17 -O3 -Wall -Wextra -I"%INCLUDE_DIR%" ^
    "%SRC_DIR%Matrix.cpp" ^
    "%SRC_DIR%Gauss.cpp" ^
    "%SRC_DIR%Bessel.cpp" ^
    "%SRC_DIR%Chebyshev.cpp" ^
    "%SRC_DIR%DifrOnLenta.cpp" ^
    "%SRC_DIR%main.cpp" ^
    -o diffraction_solver.exe
set "BUILD_STATUS=%ERRORLEVEL%"
popd > nul

if "%BUILD_STATUS%"=="0" (
    echo.
    echo Компиляция завершена
    echo Файл: build\diffraction_solver.exe
) else (
    echo.
    echo Ошибка компиляции
)
pause
goto end

:run_only
cls
echo.
echo Запуск программы
echo.
if exist "%BUILD_DIR%\diffraction_solver.exe" (
    pushd "%BUILD_DIR%" > nul
    diffraction_solver.exe
    popd > nul
    pause
) else (
    echo Ошибка: файл не найден
    echo Сначала выполните компиляцию (опция 3)
    pause
)
goto end

:invalid
echo.
echo Неверный выбор
timeout /t 2 > nul
goto menu

:end
endlocal
