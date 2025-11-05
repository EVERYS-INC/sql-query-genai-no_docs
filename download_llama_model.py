"""
Llama-3-ELYZA-JP-8Bモデルを事前にダウンロードするスクリプト
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model():
    """モデルとトークナイザーをダウンロードしてローカルに保存"""
    
    # ローカル保存先のディレクトリ
    model_dir = Path("models/llama-3-elyza-jp-8b")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Llama-3-ELYZA-JP-8Bモデルのダウンロードを開始します")
    print("モデルサイズ: 約16GB")
    print("保存先: models/llama-3-elyza-jp-8b/")
    print("=" * 60)
    
    try:
        # モデル名
        model_name = "elyza/Llama-3-ELYZA-JP-8B"
        
        # トークナイザーをダウンロード
        print("\n[1/2] トークナイザーをダウンロード中...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        # ローカルに保存
        tokenizer.save_pretrained(model_dir)
        print("✓ トークナイザーのダウンロード完了")
        
        # モデルをダウンロード
        print("\n[2/2] モデルをダウンロード中...")
        print("※ 初回は16GBのダウンロードのため、時間がかかります（20-60分程度）")
        
        # CPUでダウンロード（メモリ節約）
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
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
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {str(e)}")
        print("\n考えられる原因:")
        print("1. インターネット接続の問題")
        print("2. ディスク容量不足（20GB以上必要）")
        print("3. メモリ不足（16GB以上推奨）")
        return False

def check_existing_model():
    """既存のモデルをチェック"""
    model_dir = Path("models/llama-3-elyza-jp-8b")
    
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
        
        # モデルファイルの存在チェック（.safetensorsまたは.bin）
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
    print("Llama-3-ELYZA-JP-8Bモデル ダウンロードツール")
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
        print("app.pyを起動してLlamaモデルを選択できます。")
    else:
        print("\n❌ ダウンロードに失敗しました。")
        sys.exit(1)

if __name__ == "__main__":
    main()