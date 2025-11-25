@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ════════════════════════════════════════════════════
echo   ЗАПУСК ПРОГРАММЫ ДИФРАКЦИИ
echo ════════════════════════════════════════════════════
echo.

REM Проверяем наличие скомпилированного файла
if exist "bin\Debug\Diffraction.exe" (
    echo [✓] Найден скомпилированный файл
    echo.
    echo Запуск программы...
    start "" "bin\Debug\Diffraction.exe"
    exit /b 0
)

echo [!] Программа не скомпилирована
echo.
echo ════════════════════════════════════════════════════
echo   КОМПИЛЯЦИЯ ПРОЕКТА
echo ════════════════════════════════════════════════════
echo.

REM Создаем директории если их нет
if not exist "bin\Debug" mkdir "bin\Debug"
if not exist "obj\Debug" mkdir "obj\Debug"

REM Ищем MSBuild
set MSBUILD_PATH=
for %%v in (4.0.30319 4.5 4.6 4.7 4.8) do (
    if exist "%SystemRoot%\Microsoft.NET\Framework\%%v\MSBuild.exe" (
        set MSBUILD_PATH=%SystemRoot%\Microsoft.NET\Framework\%%v\MSBuild.exe
    )
)

if "%MSBUILD_PATH%"=="" (
    echo [✗] ОШИБКА: MSBuild не найден!
    echo.
    echo РЕШЕНИЕ:
    echo 1. Установите .NET Framework 4.7.2 или выше
    echo 2. Или используйте скрипт ОЧИСТИТЬ_И_СОБРАТЬ.bat
    echo.
    pause
    exit /b 1
)

echo Компиляция с помощью MSBuild...
echo.
"%MSBUILD_PATH%" Diffraction.csproj /p:Configuration=Debug /t:Build /verbosity:minimal /nologo

if errorlevel 1 (
    echo.
    echo ════════════════════════════════════════════════════
    echo   [✗] ОШИБКА КОМПИЛЯЦИИ!
    echo ════════════════════════════════════════════════════
    echo.
    echo ВОЗМОЖНЫЕ РЕШЕНИЯ:
    echo.
    echo 1. Переместите проект в путь БЕЗ кириллицы и скобок:
    echo    Плохо:  C:\Users\Дима\Downloads\Diffraction-main(1)\
    echo    Хорошо: C:\Projects\Diffraction\
    echo.
    echo 2. Запустите скрипт: ОЧИСТИТЬ_И_СОБРАТЬ.bat
    echo.
    echo 3. Удалите папки bin\ и obj\ вручную, затем повторите запуск
    echo.
    pause
    exit /b 1
)

echo.
echo ════════════════════════════════════════════════════
echo   [✓] КОМПИЛЯЦИЯ УСПЕШНА!
echo ════════════════════════════════════════════════════
echo.
echo Запуск программы...
start "" "bin\Debug\Diffraction.exe"
