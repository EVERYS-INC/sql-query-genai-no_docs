"""
CPU専用のLlama Client - 4GB VRAM環境用の代替実装
"""

import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

class LlamaClientCPU:
    def __init__(self):
        # ローカルモデルのパス
        self.local_model_path = Path("models/llama-3-elyza-jp-8b")
        self.device = "cpu"  # CPU固定
        self.model = None
        self.tokenizer = None
        
        print("=" * 60)
        print("CPU専用モード")
        print("GPUメモリ不足のため、CPU専用モードで動作します")
        print("処理には30-60秒かかります")
        print("=" * 60)
        
        self.load_model()
    
    def load_model(self):
        """モデルとトークナイザーを読み込む（CPU専用）"""
        try:
            if not self.local_model_path.exists():
                raise ValueError(f"モデルが見つかりません: {self.local_model_path}")
            
            model_path = str(self.local_model_path)
            
            print("トークナイザーを読み込み中...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            print("CPUモデルを読み込み中（数分かかる場合があります）...")
            
            # メモリ効率的な読み込み
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,  # CPUではfloat32を使用
                    low_cpu_mem_usage=True,  # メモリ効率化
                    trust_remote_code=True,
                    local_files_only=True
                )
            
            print("✓ モデル読み込み完了（CPUモード）")
            
            # CPUスレッド最適化
            torch.set_num_threads(os.cpu_count() or 4)
            print(f"CPUスレッド数: {torch.get_num_threads()}")
                
        except Exception as e:
            raise ValueError(f"モデル読み込みエラー: {str(e)}")
    
    def generate_sql(self, natural_language_query, table_schema, debug=False):
        # シンプルなプロンプト（トークン数削減）
        prompt = f"""SQLiteのSQLクエリを生成:

テーブル:
{table_schema[:500]}

質問: {natural_language_query}

SQL:
"""
        
        try:
            print("トークナイズ中...")
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=1024  # トークン数制限
            )
            
            print(f"入力トークン数: {len(inputs['input_ids'][0])}")
            
            # CPU用の生成設定（高速化）
            generation_config = {
                "max_new_tokens": 150,  # 短めに設定
                "temperature": 0.1,
                "do_sample": False  # 決定的生成
            }
            
            # pad_token_idとeos_token_idを設定
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            generation_config["eos_token_id"] = self.tokenizer.eos_token_id
            
            print("SQL生成中（30-60秒かかります）...")
            import time
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            
            elapsed_time = time.time() - start_time
            print(f"✅ 生成完了（{elapsed_time:.1f}秒）")
            
            # デコード
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            sql_query = response[len(prompt):].strip()
            
            # SQLクエリのクリーニング
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            if "SELECT" in sql_query.upper():
                sql_query = sql_query[sql_query.upper().find("SELECT"):]
            if ";" in sql_query:
                sql_query = sql_query[:sql_query.find(";")+1]
            
            if debug:
                debug_info = {
                    "prompt": prompt,
                    "raw_response": response,
                    "cleaned_sql": sql_query,
                    "model": "Llama-3-ELYZA-JP-8B (CPU)",
                    "device": "cpu",
                    "processing_time": f"{elapsed_time:.1f}秒"
                }
                return sql_query, debug_info
            
            return sql_query
            
        except Exception as e:
            print(f"エラー: {str(e)}")
            # フォールバック: 基本的なSQLを返す
            return "SELECT * FROM production_records ORDER BY production_date DESC LIMIT 10"
    
    def suggest_visualization(self, query, dataframe):
        """簡易的な可視化提案"""
        if len(dataframe.columns) >= 2:
            return "bar"
        return "table"