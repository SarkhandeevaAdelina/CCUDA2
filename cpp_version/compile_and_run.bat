@echo off
setlocal
chcp 65001 > nul
set "PATH=C:\msys64\mingw64\bin;%PATH%"

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build"
set "SRC_DIR=%SCRIPT_DIR%src\"
set "INCLUDE_DIR=%SCRIPT_DIR%include"

echo.
echo Компиляция и запуск
echo.

if not exist "%BUILD_DIR%" (
    echo Ошибка: папка build не найдена
    echo Ожидалось расположение: %BUILD_DIR%
    echo.
    pause
    exit /b 1
)

pushd "%BUILD_DIR%" > nul

echo Компиляция...
g++ -std=c++17 -O3 -Wall -Wextra -I"%INCLUDE_DIR%" ^
    "%SRC_DIR%Matrix.cpp" ^
    "%SRC_DIR%Gauss.cpp" ^
    "%SRC_DIR%Bessel.cpp" ^
    "%SRC_DIR%Chebyshev.cpp" ^
    "%SRC_DIR%DifrOnLenta.cpp" ^
    "%SRC_DIR%main.cpp" ^
    -o diffraction_solver.exe

set "BUILD_STATUS=%ERRORLEVEL%"

if "%BUILD_STATUS%"=="0" (
    echo.
    echo Компиляция завершена
    echo.
    echo Запуск программы
    echo.
    diffraction_solver.exe
    echo.
    echo Программа завершена
) else (
    echo.
    echo Ошибка компиляции
)

popd > nul
echo.
pause
endlocal
