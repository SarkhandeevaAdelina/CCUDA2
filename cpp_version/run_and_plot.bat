@echo off
setlocal
chcp 65001 > nul
set "PATH=C:\msys64\mingw64\bin;%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build"
set "SRC_DIR=%SCRIPT_DIR%src\"
set "INCLUDE_DIR=%SCRIPT_DIR%include"
set "EXECUTABLE=%BUILD_DIR%\diffraction_solver.exe"
set "OUTPUT_TXT=%BUILD_DIR%\output.txt"
set "TIMING_CSV=%SCRIPT_DIR%timing_data.csv"
set "TIMING_TEMP=%SCRIPT_DIR%timing_data_temp.csv"

echo.
echo Запуск с построением графиков
echo.

if not exist "%BUILD_DIR%" (
    echo Ошибка: папка build не найдена
    echo.
    pause
    exit /b 1
)

REM Проверка наличия исполняемого файла
if not exist "%EXECUTABLE%" (
    echo Исполняемый файл не найден, компиляция...
    pushd "%BUILD_DIR%" > nul
    g++ -std=c++17 -O3 -Wall -Wextra -I"%INCLUDE_DIR%" ^
        "%SRC_DIR%Matrix.cpp" ^
        "%SRC_DIR%Gauss.cpp" ^
        "%SRC_DIR%Bessel.cpp" ^
        "%SRC_DIR%Chebyshev.cpp" ^
        "%SRC_DIR%DifrOnLenta.cpp" ^
        "%SRC_DIR%main.cpp" ^
        -o diffraction_solver.exe
    set "COMPILE_STATUS=%ERRORLEVEL%"
    popd > nul
    
    if not "%COMPILE_STATUS%"=="0" (
        echo Ошибка компиляции
        pause
        exit /b 1
    )
)

echo Запуск программы...
pushd "%BUILD_DIR%" > nul
diffraction_solver.exe > "%OUTPUT_TXT%"
type "%OUTPUT_TXT%"
popd > nul

echo.
echo Извлечение CSV данных...
findstr /R "^[0-9][0-9]*," "%OUTPUT_TXT%" > "%TIMING_CSV%"

REM Проверка данных
set "size=0"
for %%A in ("%TIMING_CSV%") do set size=%%~zA
if "%size%"=="0" (
    echo CSV данные не найдены
    pause
    exit /b 1
)

REM Добавление заголовка
echo N,MatrixTime_ms,SolveTime_ms,TotalTime_ms > "%TIMING_TEMP%"
type "%TIMING_CSV%" >> "%TIMING_TEMP%"
move /Y "%TIMING_TEMP%" "%TIMING_CSV%" > nul

echo CSV данные сохранены: %TIMING_CSV%
echo.

pushd "%SCRIPT_DIR%" > nul
echo Построение графиков...
python plot_timing.py "%TIMING_CSV%"
set "PLOT_STATUS=%ERRORLEVEL%"
popd > nul

if "%PLOT_STATUS%"=="0" (
    echo.
    echo Графики сохранены:
    echo - timing_analysis.png
    echo - complexity_analysis.png
    echo.
    echo Расположение: %SCRIPT_DIR%
    
    REM Открыть графики
    if exist "%SCRIPT_DIR%timing_analysis.png" start "" "%SCRIPT_DIR%timing_analysis.png"
    if exist "%SCRIPT_DIR%complexity_analysis.png" (
        timeout /t 1 > nul
        start "" "%SCRIPT_DIR%complexity_analysis.png"
    )
) else (
    echo Ошибка построения графиков
)

echo.
pause
endlocal
