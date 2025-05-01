@echo off
echo ===== Audio Classification Docker Helper =====

REM Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Docker is not installed. Please install Docker Desktop first.
    exit /b 1
)

REM Create necessary directories
echo Creating necessary directories...
if not exist evaluation mkdir evaluation
if not exist models mkdir models
if not exist clean mkdir clean
if not exist plots mkdir plots
if not exist output mkdir output

REM Parse command line arguments
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=run

if "%COMMAND%"=="run" (
    call :run_project
) else if "%COMMAND%"=="build" (
    call :build_project
) else if "%COMMAND%"=="script" (
    call :run_script %2
) else if "%COMMAND%"=="clean" (
    call :clean_docker
) else if "%COMMAND%"=="help" (
    call :show_help
) else (
    echo Unknown command: %COMMAND%
    call :show_help
    exit /b 1
)

exit /b 0

:run_project
echo Building and running the Docker container...
docker-compose build
docker-compose up
exit /b 0

:build_project
echo Building the Docker image...
docker-compose build
exit /b 0

:run_script
set SCRIPT=%1
if "%SCRIPT%"=="" (
    echo Please specify a script to run.
    exit /b 1
)

echo Running %SCRIPT%...
docker-compose run --rm audio-tools %SCRIPT%
exit /b 0

:clean_docker
echo Removing Docker containers and images...
docker-compose down
docker rmi audio-classification:latest
echo Cleanup complete.
exit /b 0

:show_help
echo Audio Classification Docker Helper
echo.
echo Usage:
echo   run_docker.bat [command]
echo.
echo Commands:
echo   run             Run the complete workflow (default)
echo   build           Build the Docker image
echo   script ^<name^>   Run a specific script (e.g., display_plots.py)
echo   clean           Remove all Docker containers and images
echo   help            Show this help message
echo.
echo Examples:
echo   run_docker.bat run
echo   run_docker.bat script display_plots.py
echo   run_docker.bat script model.py
exit /b 0
