@echo off
setlocal
chcp 65001 > nul

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build"
set "EXECUTABLE=%BUILD_DIR%\diffraction_solver.exe"

echo.
echo Запуск программы
echo.

if exist "%EXECUTABLE%" (
    pushd "%BUILD_DIR%" > nul
    diffraction_solver.exe
    popd > nul
    echo.
    pause
) else (
    echo Ошибка: файл не найден
    echo Сначала скомпилируйте проект
    echo.
    pause
)

endlocal
