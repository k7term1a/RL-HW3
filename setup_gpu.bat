@echo off
echo ============================================================
echo  RL HW3 - GPU Environment Setup (GTX 1650 Ti / CUDA 12.6)
echo ============================================================

echo.
echo [1/3] Installing all dependencies with uv...
uv sync
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: uv sync failed.
    pause
    exit /b 1
)

echo.
echo [2/3] Replacing CPU torch with CUDA 12.6 build...
uv pip install torch torchvision ^
    --index-url https://download.pytorch.org/whl/cu126 ^
    --reinstall --no-deps
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CUDA torch installation failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Verifying GPU availability...
uv run --frozen python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"

echo.
echo ============================================================
echo  Setup complete!  Use the following commands to train:
echo.
echo    uv run --frozen python hw3_1_naive_dqn.py
echo    uv run --frozen python hw3_2_enhanced_dqn.py
echo    uv run --frozen python hw3_3_lightning_dqn.py
echo    uv run --frozen python hw3_4_rainbow_dqn.py
echo.
echo  --frozen prevents uv from re-syncing (which would revert
echo  torch back to the CPU build from PyPI).
echo ============================================================
pause
