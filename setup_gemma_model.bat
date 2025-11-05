@echo off
chcp 65001 > nul
echo ========================================
echo Gemma-2-2b-jpn-it モデル セットアップ
echo ========================================
echo.

REM Python存在確認
python --version > nul 2>&1
if errorlevel 1 (
    echo [エラー] Pythonがインストールされていません
    echo https://www.python.org/downloads/ からPython 3.11をインストールしてください
    pause
    exit /b 1
)

REM 仮想環境の確認
if not exist "venv\Scripts\activate.bat" (
    echo [エラー] 仮想環境が見つかりません
    echo 先に setup_windows.bat を実行してください
    pause
    exit /b 1
)

echo 仮想環境を有効化中...
call venv\Scripts\activate.bat

echo.
echo ========================================
echo Gemma-2-2b-jpn-it について
echo ========================================
echo ✨ 軽量モデル: 2Bパラメータ（Llama-3の1/4）
echo ⚡ 高速処理: Llama-3の約3倍速
echo 💾 必要容量: 約5GB（ダウンロード）
echo 🎮 必要VRAM: 4GB以上（RTX 3050でも快適）
echo 🌐 オフライン: 完全対応
echo ========================================
echo.

REM Hugging Face トークンの確認
echo 📝 Gemmaモデルには利用規約への同意が必要です
echo.
echo 手順:
echo 1. https://huggingface.co/google/gemma-2-2b-jpn-it にアクセス
echo 2. 利用規約に同意（Agree and access repository）
echo 3. https://huggingface.co/settings/tokens でトークン取得
echo.

set /p HF_TOKEN=Hugging Faceトークンを入力（スキップする場合はEnter）: 

if not "%HF_TOKEN%"=="" (
    echo.
    echo トークンを環境変数に設定中...
    setx HF_TOKEN "%HF_TOKEN%" > nul 2>&1
    set HF_TOKEN=%HF_TOKEN%
    echo ✅ トークン設定完了
)

echo.
echo ========================================
echo 必要なパッケージを確認中...
echo ========================================

REM transformersのインストール確認
pip show transformers > nul 2>&1
if errorlevel 1 (
    echo transformersをインストール中...
    pip install transformers accelerate sentencepiece protobuf
)

REM torchのインストール確認
pip show torch > nul 2>&1
if errorlevel 1 (
    echo PyTorchをインストール中...
    
    REM GPU確認
    nvidia-smi > nul 2>&1
    if errorlevel 1 (
        echo CPUモード用PyTorchをインストール...
        pip install torch torchvision torchaudio
    ) else (
        echo GPU対応PyTorchをインストール...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    )
)

echo.
echo ========================================
echo モデルのダウンロード
echo ========================================
echo.

REM モデルディレクトリの確認
if exist "models\gemma-2-2b-jpn-it\config.json" (
    echo ✅ モデルは既にダウンロード済みです
    echo.
    set /p REDOWNLOAD=再ダウンロードしますか？ (y/N): 
    if /i not "%REDOWNLOAD%"=="y" (
        goto :check_gpu
    )
)

echo Gemmaモデルをダウンロード中...
echo ※ 約5GBのダウンロードのため、5-15分程度かかります
echo.

python download_gemma_model.py
if errorlevel 1 (
    echo.
    echo ⚠️ ダウンロードに失敗しました
    echo.
    echo 考えられる原因:
    echo 1. Hugging Faceトークンが無効
    echo 2. 利用規約に同意していない
    echo 3. インターネット接続の問題
    echo.
    pause
    exit /b 1
)

:check_gpu
echo.
echo ========================================
echo GPU環境の確認
echo ========================================
python check_gpu.py

echo.
echo ========================================
echo ✅ セットアップ完了！
echo ========================================
echo.
echo 使い方:
echo 1. streamlit run app.py でアプリを起動
echo 2. サイドバーで「💎 Gemma-2-2b-jpn-it」を選択
echo 3. 「モデルを読み込む」をクリック
echo 4. 自然言語でクエリを実行
echo.
echo 特徴:
echo - 軽量（2Bパラメータ）
echo - 高速（Llama-3の約3倍速）
echo - 4GB VRAMで快適動作
echo.
pause