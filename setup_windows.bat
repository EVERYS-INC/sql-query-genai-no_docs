@echo off
echo ========================================
echo Windows 11 セットアップスクリプト
echo ========================================
echo.

REM UTF-8エンコーディング設定
chcp 65001 > nul
set PYTHONUTF8=1

echo [1/6] Python環境を確認中...
python --version
if errorlevel 1 (
    echo ERROR: Pythonがインストールされていません
    echo https://www.python.org/downloads/ からインストールしてください
    pause
    exit /b 1
)

echo.
echo [2/6] 仮想環境を作成中...
if not exist venv (
    python -m venv venv
    echo 仮想環境を作成しました
) else (
    echo 仮想環境は既に存在します
)

echo.
echo [3/6] 仮想環境を有効化中...
call venv\Scripts\activate.bat

echo.
echo [4/6] pipをアップグレード中...
python -m pip install --upgrade pip

echo.
echo [5/6] 依存関係をインストール中...
pip install -r requirements_windows.txt

echo.
echo [6/6] サンプルデータベースを作成中...
if not exist database\factory_data.db (
    cd database
    python create_factory_db.py
    cd ..
    echo データベースを作成しました
) else (
    echo データベースは既に存在します
)

echo.
echo ========================================
echo セットアップが完了しました！
echo.
echo 次のコマンドでアプリケーションを起動できます:
echo   streamlit run app.py
echo.
echo 注意: .envファイルにAzure OpenAIの設定を忘れずに！
echo ========================================
pause