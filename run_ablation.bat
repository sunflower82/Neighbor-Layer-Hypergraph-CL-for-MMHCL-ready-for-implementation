@echo off
REM ============================================================
REM  MMHCL+ Rev 5.2 Q1-style Ablation Study launcher (Windows).
REM  Activates the rtx5090_dl conda env and kicks off the full
REM  15 variant x 3 seed sweep on Baby using run_ablation.py.
REM
REM  Override behaviour:
REM    set ABLATION_VARIANTS=A0_full,A7_ego_final         (subset)
REM    set ABLATION_SEEDS=3                               (n seeds)
REM    set ABLATION_EPOCHS=250                            (per run)
REM    set ABLATION_DATASET=Baby
REM
REM  Smoke test example:
REM    set ABLATION_VARIANTS=A0_full
REM    set ABLATION_SEEDS=1
REM    set ABLATION_EPOCHS=10
REM    run_ablation.bat
REM ============================================================

setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0"

set CONDA_ENV=rtx5090_dl
set CONDA_EXE=C:\ProgramData\anaconda3\Scripts\conda.exe

if "%ABLATION_VARIANTS%"=="" (
    set "ABLATION_VARIANTS=A0_full,A1_no_nlcl,A2_no_svd,A3_small_proj,A4_no_ramp,A5_no_delay,A6_no_dirichlet,A7_ego_final,A8_no_cross,B1_g1,B2_g2,B3_g3,C1_uncertainty,C2_gradnorm,C3_fixed"
)
if "%ABLATION_SEEDS%"==""   set ABLATION_SEEDS=3
if "%ABLATION_EPOCHS%"==""  set ABLATION_EPOCHS=250
if "%ABLATION_DATASET%"=="" set ABLATION_DATASET=Baby
if "%ABLATION_GPU%"==""     set ABLATION_GPU=0

echo ========================================================================
echo MMHCL+ Rev 5.2 Ablation Sweep
echo ========================================================================
echo   dataset  : %ABLATION_DATASET%
echo   gpu      : %ABLATION_GPU%
echo   variants : %ABLATION_VARIANTS%
echo   seeds    : %ABLATION_SEEDS%
echo   epochs   : %ABLATION_EPOCHS%
echo ========================================================================

call "%CONDA_EXE%" run -n %CONDA_ENV% --no-capture-output python codes\run_ablation.py ^
    --dataset %ABLATION_DATASET% ^
    --gpu %ABLATION_GPU% ^
    --variants %ABLATION_VARIANTS% ^
    --seeds %ABLATION_SEEDS% ^
    --epochs %ABLATION_EPOCHS% ^
    --out-dir "%CD%\ablation_outputs"

endlocal
echo.
echo Sweep finished. Results in ablation_outputs\.
pause
