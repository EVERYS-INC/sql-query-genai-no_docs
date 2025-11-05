@echo off
echo ====================================================
echo Llama-3-ELYZA-JP-8B モデルセットアップ
echo ====================================================
echo.

REM 仮想環境のアクティベート
if exist venv\Scripts\activate.bat (
    echo 仮想環境をアクティベート中...
    call venv\Scripts\activate.bat
) else (
    echo 警告: 仮想環境が見つかりません
    echo python環境を直接使用します
)

echo.
echo モデルのダウンロードを開始します...
echo ※ 約16GBのダウンロードのため、20-60分程度かかります
echo.

python download_llama_model.py

if %errorlevel% neq 0 (
    echo.
    echo エラー: モデルのダウンロードに失敗しました
    pause
    exit /b 1
)

echo.
echo ====================================================
echo セットアップ完了！
echo.
echo 次の手順:
echo 1. streamlit run app.py でアプリを起動
echo 2. サイドバーで「Llama-3-ELYZA-JP-8B」を選択
echo 3. 「Llamaモデルを読み込む」ボタンをクリック
echo ====================================================
pause