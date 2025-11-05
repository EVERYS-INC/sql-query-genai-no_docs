"""
Gemma-2-2b-jpn-it Client - 軽量で高速な日本語モデル
4GB VRAMでも快適に動作
"""

import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time
from dotenv import load_dotenv

# .envファイルから環境変数を読み込み
load_dotenv()

class GemmaClient:
    def __init__(self, force_cpu=False):
        # ローカルモデルのパス
        self.local_model_path = Path("models/gemma-2-2b-jpn-it")
        # オンラインモデル名（フォールバック用）
        self.online_model_name = "google/gemma-2-2b-jpn-it"
        
        # GPU/CPUの選択
        if force_cpu:
            self.device = "cpu"
            print("💻 CPUモードで動作します")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"⚡ GPU検出: {gpu_name} ({vram:.1f}GB)")
                print("✨ Gemma-2-2bは軽量なため、4GB VRAMでも快適に動作します")
            else:
                print("💻 GPUが利用できないため、CPUモードで動作します")
        
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """モデルとトークナイザーを読み込む"""
        try:
            # ローカルモデルが存在するかチェック
            if self.local_model_path.exists() and (self.local_model_path / "config.json").exists():
                print(f"ローカルモデルを読み込み中: {self.local_model_path}")
                model_path = str(self.local_model_path)
                local_mode = True
                hf_token = None  # ローカルモデルの場合はトークン不要
            else:
                # オンラインからダウンロードを試みる
                print(f"ローカルモデルが見つかりません。オンラインからダウンロードを試みます...")
                
                # .envファイルからHF_TOKENを取得
                hf_token = os.environ.get("HF_TOKEN", None)
                if hf_token:
                    print("✓ Hugging Faceトークンを.envファイルから取得しました")
                    model_path = self.online_model_name
                    local_mode = False
                else:
                    raise ValueError(
                        f"ローカルモデルが見つかりません: {self.local_model_path.absolute()}\n"
                        "また、HF_TOKENが.envファイルに設定されていません。\n"
                        "以下のいずれかを実行してください：\n"
                        "1. 'python download_gemma_model.py' を実行してモデルをダウンロード\n"
                        "2. .envファイルにHF_TOKEN=your-token-hereを追加"
                    )
            
            # トークナイザーを読み込む
            print("トークナイザーを読み込み中...")
            tokenizer_kwargs = {
                "trust_remote_code": True,
                "local_files_only": local_mode
            }
            if not local_mode and hf_token:
                tokenizer_kwargs["token"] = hf_token
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                **tokenizer_kwargs
            )
            
            # パッドトークンを設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # モデルを読み込む
            if self.device == "cuda":
                print("GPUモデルを読み込み中...")
                
                # モデル読み込み用の共通kwargs
                model_kwargs = {
                    "trust_remote_code": True,
                    "local_files_only": local_mode
                }
                if not local_mode and hf_token:
                    model_kwargs["token"] = hf_token
                
                # 4GB VRAMでも動作するように設定
                if torch.cuda.get_device_properties(0).total_memory < 6 * 1024**3:  # 6GB未満
                    print("⚠️ VRAM容量が限定的なため、8bit量子化を使用します")
                    try:
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_enable_fp32_cpu_offload=True
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            quantization_config=quantization_config,
                            device_map="auto",
                            **model_kwargs
                        )
                        print("✓ 8bit量子化モデル読み込み完了")
                    except (ImportError, Exception) as e:
                        print(f"⚠️ 8bit量子化での読み込みに失敗: {str(e)}")
                        print("通常モードで読み込みます...")
                        model_kwargs["torch_dtype"] = torch.float16
                        model_kwargs["low_cpu_mem_usage"] = False  # meta deviceエラーを回避
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            **model_kwargs
                        )
                        self.model = self.model.to(self.device)
                else:
                    # 十分なVRAMがある場合
                    model_kwargs["torch_dtype"] = torch.float16
                    model_kwargs["low_cpu_mem_usage"] = False  # meta deviceエラーを回避
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
                    self.model = self.model.to(self.device)
                
                print(f"✓ GPUモデル読み込み完了")
                print(f"   GPU使用メモリ: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            else:
                # CPUの場合
                print("CPUモデルを読み込み中...")
                cpu_kwargs = {
                    "torch_dtype": torch.float32,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                    "local_files_only": local_mode
                }
                if not local_mode and hf_token:
                    cpu_kwargs["token"] = hf_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **cpu_kwargs
                )
                self.model = self.model.to(self.device)
                print("✓ CPUモデル読み込み完了")
                
        except Exception as e:
            raise ValueError(f"Gemma-2-2b-jpn-itモデルの読み込みエラー: {str(e)}")
    
    def generate_sql(self, natural_language_query, table_schema, debug=False):
        """SQLクエリを生成"""
        
        # Gemma用のプロンプトフォーマット（シンプル版）
        prompt = f"""<bos><start_of_turn>user
SQLiteのテーブルに対して、以下の質問をSQLクエリに変換してください。

テーブル構造:
{table_schema}

質問: {natural_language_query}

注意:
- SQLiteの日付関数を使用
- 今月: date('now', 'start of month') から date('now')
- 必要なJOINのみ使用
- SELECT文から始まるSQLクエリを返す<end_of_turn>
<start_of_turn>model
SELECT"""
        
        try:
            # トークナイズ
            print("プロンプトをトークナイズ中...")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # デバイスに移動
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            print(f"入力トークン数: {len(inputs['input_ids'][0])}")
            
            # 生成設定（より安定した出力のため調整）
            generation_config = {
                "max_new_tokens": 300,  # SQLクエリ用に増加
                "temperature": 0.05,     # より決定的な出力
                "do_sample": True,
                "top_p": 0.9,           # より保守的な選択
                "top_k": 30,            # トップ候補を絞る
                "repetition_penalty": 1.05  # 軽い繰り返し抑制
            }
            
            if self.device == "cuda":
                print("⚡ GPU高速生成中...")
            else:
                print("💻 CPU生成中（5-15秒）...")
            
            start_time = time.time()
            
            with torch.no_grad():
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                outputs = self.model.generate(
                    **inputs,
                    **generation_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            elapsed_time = time.time() - start_time
            print(f"✅ SQL生成完了 (処理時間: {elapsed_time:.1f}秒)")
            
            # デコード
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # モデルの回答部分を抽出（改善版）
            # "SELECT"で終わるプロンプトを考慮
            if "SELECT" in response:
                # 最後のSELECTの位置を見つける
                select_positions = [i for i in range(len(response)) if response[i:].startswith("SELECT")]
                if select_positions:
                    # 最も適切なSELECT位置を選ぶ（プロンプトの後のもの）
                    prompt_select_count = prompt.count("SELECT")
                    if len(select_positions) > prompt_select_count:
                        sql_query = "SELECT" + response[select_positions[prompt_select_count] + 6:]
                    else:
                        sql_query = response[select_positions[-1]:]
                else:
                    sql_query = response.strip()
            else:
                # フォールバック処理
                if "model" in response:
                    model_idx = response.rfind("model")
                    if model_idx != -1:
                        sql_query = response[model_idx + 5:].strip()
                    else:
                        sql_query = response.strip()
                else:
                    sql_query = response[len(prompt):].strip()
            
            
            # コードブロックの処理を改善
            if "```" in sql_query:
                # ```sql または ``` の処理
                lines = sql_query.split('\n')
                in_code_block = False
                sql_lines = []
                
                for line in lines:
                    if line.strip().startswith("```"):
                        if not in_code_block:
                            # コードブロック開始
                            in_code_block = True
                        else:
                            # コードブロック終了
                            break
                    elif in_code_block:
                        # コードブロック内のコンテンツ
                        sql_lines.append(line)
                
                if sql_lines:
                    sql_query = '\n'.join(sql_lines).strip()
                else:
                    # 古い方法にフォールバック
                    if "```sql" in sql_query.lower():
                        start_idx = sql_query.lower().find("```sql") + 6
                    else:
                        start_idx = sql_query.find("```") + 3
                        # 改行までスキップ
                        newline_idx = sql_query.find("\n", start_idx)
                        if newline_idx != -1 and newline_idx - start_idx < 10:
                            start_idx = newline_idx + 1
                    
                    end_idx = sql_query.find("```", start_idx)
                    if end_idx != -1:
                        sql_query = sql_query[start_idx:end_idx].strip()
                    else:
                        sql_query = sql_query[start_idx:].strip()
            
            # SQLクエリが適切に始まっているか確認
            sql_upper = sql_query.upper()
            if not sql_upper.startswith(("SELECT", "INSERT", "UPDATE", "DELETE", "WITH")):
                # SELECT文を探す
                if "SELECT" in sql_upper:
                    sql_start = sql_upper.find("SELECT")
                    sql_query = sql_query[sql_start:]
            
            # セミコロンで終端
            if ";" in sql_query:
                sql_query = sql_query[:sql_query.find(";")+1]
            
            # 最終的なクリーンアップ
            sql_query = sql_query.strip()
            
            if debug:
                debug_info = {
                    "prompt": prompt,
                    "raw_response": response,
                    "cleaned_sql": sql_query,
                    "model": "Gemma-2-2b-jpn-it",
                    "device": self.device,
                    "processing_time": f"{elapsed_time:.1f}秒",
                    "tokens_used": len(inputs['input_ids'][0]) + len(outputs[0])
                }
                return sql_query, debug_info
            
            return sql_query
            
        except Exception as e:
            raise Exception(f"SQL生成エラー: {str(e)}")
    
    def suggest_visualization(self, query, dataframe):
        """可視化タイプを提案"""
        columns = list(dataframe.columns)
        
        # シンプルなルールベース提案（高速化のため）
        if any("date" in col.lower() or "日" in col for col in columns):
            return "line"
        elif len(columns) == 2 and len(dataframe) <= 10:
            return "pie"
        elif any("rate" in col.lower() or "率" in col for col in columns):
            return "bar"
        else:
            return "bar"