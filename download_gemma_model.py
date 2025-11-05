"""
Gemma-2-2b-jpn-itモデルを事前にダウンロードするスクリプト
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model():
    """モデルとトークナイザーをダウンロードしてローカルに保存"""
    
    # ローカル保存先のディレクトリ
    model_dir = Path("models/gemma-2-2b-jpn-it")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Gemma-2-2b-jpn-itモデルのダウンロードを開始します")
    print("モデルサイズ: 約5GB（Llama-3の1/3）")
    print("保存先: models/gemma-2-2b-jpn-it/")
    print("=" * 60)
    
    try:
        # モデル名
        model_name = "google/gemma-2-2b-jpn-it"
        
        # Hugging Faceのアクセストークンが必要な場合の注意
        print("\n注意: Gemmaモデルは利用規約への同意が必要です")
        print("1. https://huggingface.co/google/gemma-2-2b-jpn-it にアクセス")
        print("2. 利用規約に同意")
        print("3. Hugging Faceにログイン")
        print("")
        
        # トークンの確認 - .envファイルから優先的に取得
        from dotenv import load_dotenv
        load_dotenv()
        
        hf_token = os.environ.get("HF_TOKEN", None)
        if not hf_token:
            print("環境変数 HF_TOKEN が設定されていません")
            print("Hugging Faceのアクセストークンを入力してください")
            print("（https://huggingface.co/settings/tokens から取得）")
            hf_token = input("トークン: ").strip()
            if not hf_token:
                print("トークンが入力されませんでした。")
                print("公開モデルとしてダウンロードを試みます...")
                hf_token = None
        else:
            print(f"✓ Hugging Faceトークンを.envファイルから取得しました")
        
        # トークナイザーをダウンロード
        print("\n[1/2] トークナイザーをダウンロード中...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True
        )
        # ローカルに保存
        tokenizer.save_pretrained(model_dir)
        print("✓ トークナイザーのダウンロード完了")
        
        # モデルをダウンロード
        print("\n[2/2] モデルをダウンロード中...")
        print("※ 約5GBのダウンロードのため、5-15分程度かかります")
        
        # CPUでダウンロード（メモリ節約）
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # ローカルに保存
        model.save_pretrained(model_dir)
        print("✓ モデルのダウンロード完了")
        
        print("\n" + "=" * 60)
        print("✅ ダウンロード完了！")
        print(f"モデルは {model_dir.absolute()} に保存されました")
        print("これでオフライン環境でも使用可能です")
        print("\n特徴:")
        print("- 軽量（2Bパラメータ）")
        print("- 高速（Llama-3の約3倍速）")
        print("- 4GB VRAMで快適動作")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {str(e)}")
        print("\n考えられる原因:")
        print("1. Hugging Faceのトークンが無効")
        print("2. 利用規約に同意していない")
        print("3. インターネット接続の問題")
        print("4. ディスク容量不足（5GB以上必要）")
        return False

def check_existing_model():
    """既存のモデルをチェック"""
    model_dir = Path("models/gemma-2-2b-jpn-it")
    
    if model_dir.exists():
        # 必要なファイルが存在するかチェック
        required_files = [
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json"
        ]
        
        missing_files = []
        for file in required_files:
            if not (model_dir / file).exists():
                missing_files.append(file)
        
        # モデルファイルの存在チェック
        model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
        
        if not model_files:
            print("⚠️ モデルファイルが見つかりません")
            return False
        
        if missing_files:
            print(f"⚠️ 以下のファイルが不足しています: {', '.join(missing_files)}")
            return False
        
        print("✅ モデルは既にダウンロード済みです")
        print(f"   場所: {model_dir.absolute()}")
        
        # ファイルサイズを確認
        total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        print(f"   サイズ: {total_size / (1024**3):.2f} GB")
        
        return True
    
    return False

def main():
    print("Gemma-2-2b-jpn-itモデル ダウンロードツール")
    print("=" * 60)
    
    # 既存モデルのチェック
    if check_existing_model():
        print("\nモデルは既にダウンロード済みです。")
        response = input("再ダウンロードしますか？ (y/N): ")
        if response.lower() != 'y':
            print("ダウンロードをスキップしました。")
            return
    
    # ダウンロード実行
    print("\nダウンロードを開始します...")
    if download_model():
        print("\n✅ セットアップ完了！")
        print("app.pyを起動してGemmaモデルを選択できます。")
    else:
        print("\n❌ ダウンロードに失敗しました。")
        print("\nヒント:")
        print("1. HF_TOKEN環境変数を設定:")
        print('   set HF_TOKEN="your-token-here"')
        print("2. または、スクリプト実行時に入力")
        sys.exit(1)

if __name__ == "__main__":
    main()