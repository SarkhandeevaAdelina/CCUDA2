@echo off
setlocal
chcp 65001 > nul
set "PATH=C:\msys64\mingw64\bin;%PATH%"

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build"
set "SRC_DIR=%SCRIPT_DIR%src\"
set "INCLUDE_DIR=%SCRIPT_DIR%include"

echo.
echo Компиляция проекта
echo.

if not exist "%BUILD_DIR%" (
    echo Ошибка: папка build не найдена
    echo Ожидалось расположение: %BUILD_DIR%
    echo.
    pause
    exit /b 1
)

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
    echo Компиляция завершена
    echo Файл: build\diffraction_solver.exe
) else (
    echo Ошибка компиляции
)

echo.
pause
endlocal
