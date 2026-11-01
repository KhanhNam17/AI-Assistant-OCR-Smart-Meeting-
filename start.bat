@echo off
chcp 65001 >nul
echo ===================================================
echo   Khoi dong Tro ly Cuoc hop Thong minh (Offline)
echo ===================================================

:: Di chuyen ve dung thu muc chua project
cd /d "%~dp0"

:: Kiem tra neu dang trong virtualenv
if defined VIRTUAL_ENV (
    echo [OK] Su dung Virtualenv: %VIRTUAL_ENV%
    streamlit run ui/app.py --server.port 8501
    goto end
)

:: Kiem tra neu dang trong Conda env
if defined CONDA_DEFAULT_ENV (
    echo [OK] Su dung Conda env: %CONDA_DEFAULT_ENV%
    streamlit run ui/app.py --server.port 8501
    goto end
)

:: Chay truc tiep bang streamlit trong PATH
echo [INFO] Dang khoi chay Streamlit tren cong 8501...
streamlit run ui/app.py --server.port 8501

:end
pause
