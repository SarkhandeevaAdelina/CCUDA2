@echo off
chcp 65001 > nul
echo ════════════════════════════════════════════════════
echo   ОЧИСТКА И СБОРКА ПРОЕКТА
echo ════════════════════════════════════════════════════
echo.

echo [1/3] Очистка временных файлов...
if exist bin rmdir /s /q bin
if exist obj rmdir /s /q obj
if exist .vs rmdir /s /q .vs
echo      ✓ Временные файлы удалены
echo.

echo [2/3] Создание директорий...
mkdir bin\Debug 2>nul
mkdir obj\Debug 2>nul
echo      ✓ Директории созданы
echo.

echo [3/3] Компиляция проекта...
echo.
"%SystemRoot%\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe" Diffraction.csproj /p:Configuration=Debug /p:Platform=AnyCPU /t:Clean,Build /verbosity:minimal

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ════════════════════════════════════════════════════
    echo   ✓ КОМПИЛЯЦИЯ УСПЕШНА!
    echo ════════════════════════════════════════════════════
    echo.
    echo Запуск программы...
    echo.
    start "" "bin\Debug\Diffraction.exe"
) else (
    echo.
    echo ════════════════════════════════════════════════════
    echo   ✗ ОШИБКА КОМПИЛЯЦИИ!
    echo ════════════════════════════════════════════════════
    echo.
    echo РЕШЕНИЕ: Переместите проект в путь БЕЗ кириллицы:
    echo   Например: C:\Projects\Diffraction\
    echo.
    echo Текущий путь содержит кириллицу или спецсимволы.
    echo.
)

echo.
pause

