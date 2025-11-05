import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LlamaClient:
    def __init__(self, force_cpu=False):
        # ローカルモデルのパス
        self.local_model_path = Path("models/llama-3-elyza-jp-8b")
        # オンラインモデル名（フォールバック用）
        self.online_model_name = "elyza/Llama-3-ELYZA-JP-8B"
        
        # GPU/CPUの選択
        if force_cpu:
            self.device = "cpu"
            print("💻 CPUモードで動作します")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                print(f"⚡ GPUモードで動作します (GPU: {torch.cuda.get_device_name(0)})")
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
            else:
                raise ValueError(
                    f"ローカルモデルが見つかりません: {self.local_model_path.absolute()}\n"
                    "先に 'python download_llama_model.py' を実行してモデルをダウンロードしてください。"
                )
            
            # トークナイザーを読み込む
            print("トークナイザーを読み込み中...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=local_mode
            )
            
            # メモリ効率のため、適切な設定でモデルを読み込む
            if self.device == "cuda":
                print("GPUモデルを読み込み中...")
                print(f"利用可能なGPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                
                # 4GB VRAMの場合は8bit量子化を試みる
                if torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3:  # 8GB未満
                    print("⚠️ GPU VRAMが限定的なため、8bit量子化を使用します")
                    try:
                        # bitsandbytesをインポート（インストール済みの場合）
                        import bitsandbytes as bnb
                        
                        # カスタムデバイスマップを作成
                        # 重要なレイヤーのみGPUに配置
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            load_in_8bit=True,
                            device_map="auto",
                            trust_remote_code=True,
                            local_files_only=local_mode,
                            llm_int8_enable_fp32_cpu_offload=True,  # CPUオフロードを有効化
                            max_memory={0: "3.5GB", "cpu": "20GB"}  # GPU/CPUメモリ分割
                        )
                        print("✓ 8bit量子化モデル読み込み完了（CPU/GPUハイブリッド）")
                    except (ImportError, Exception) as e:
                        print(f"⚠️ 8bit量子化エラー: {str(e)}")
                        print("⚠️ FP16モードでCPU/GPU分割を試みます")
                        
                        # FP16でCPU/GPU分割
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            trust_remote_code=True,
                            local_files_only=local_mode,
                            offload_folder="offload",  # ディスクオフロード用フォルダ
                            offload_state_dict=True,  # 状態辞書をオフロード
                            max_memory={0: "3GB", "cpu": "16GB"}  # GPU/CPU分割
                        )
                else:
                    # 十分なVRAMがある場合
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        local_files_only=local_mode
                    )
            else:
                # CPUの場合
                print("CPUモデルを読み込み中...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    local_files_only=local_mode
                )
                self.model = self.model.to(self.device)
            
            # メモリ使用状況を表示
            if self.device == "cuda":
                print(f"✓ モデル読み込み完了")
                print(f"   GPU使用メモリ: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"   GPU予約メモリ: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            else:
                print(f"✓ モデル読み込み完了 (device: {self.device})")
                
        except Exception as e:
            raise ValueError(f"Llama-3-ELYZA-JP-8Bモデルの読み込みエラー: {str(e)}")
    
    def generate_sql(self, natural_language_query, table_schema, debug=False):
        prompt = f"""
あなたは製造業の工場データベースに精通したSQL専門家です。
**重要: このデータベースはSQLiteを使用しています。SQLite固有の構文を使用してください。**

テーブル構造:
{table_schema}

ユーザーの問い合わせ:
{natural_language_query}

以下の点に注意してください：
1. 実行可能な正確なSQLクエリのみを返してください
2. 説明やコメントは含めないでください
3. 日付は'YYYY-MM-DD'形式で扱ってください
4. 集計する場合は適切なGROUP BYを使用してください
5. 結果は見やすいようにORDER BYで並び替えてください

SQLite固有の注意事項：
- DATE_TRUNC関数は使用できません。代わりにstrftime()を使用してください
- INTERVAL演算子は使用できません。代わりにdate()関数を使用してください
- CURRENT_DATEの代わりにdate('now')を使用してください
- 今月の範囲: strftime('%Y-%m-01', 'now') から strftime('%Y-%m-01', 'now', '+1 month')
- 今週の範囲: date('now', 'weekday 0', '-7 days') から date('now', 'weekday 0')
- 今日: date('now')

SQLクエリ:
"""
        
        try:
            # プロンプトをトークナイズ
            print("プロンプトをトークナイズ中...")
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # 入力をデバイスに移動（メモリ効率のため段階的に）
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            print(f"入力トークン数: {len(inputs['input_ids'][0])}")
            
            # デバイスに応じた生成設定
            if self.device == "cuda":
                generation_config = {
                    "max_new_tokens": 500,
                    "temperature": 0.1,
                    "do_sample": True,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repetition_penalty": 1.1,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                print("⚡ GPUでSQL生成中...")
            else:
                generation_config = {
                    "max_new_tokens": 300,  # CPU用にトークン数を調整
                    "temperature": 0.1,
                    "do_sample": False,  # 決定的な生成（高速化）
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                print("💻 CPUでSQL生成中... (約30-60秒かかります)")
            
            # SQLクエリを生成
            import time
            start_time = time.time()
            
            with torch.no_grad():
                if self.device == "cpu":
                    # CPUスレッド数を最適化
                    torch.set_num_threads(8)  # CPUコア数に応じて調整
                
                # GPUメモリをクリア
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    print(f"生成開始前のGPUメモリ: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
                # ストリーミング生成（進捗を表示）
                print("SQL生成中...", end="", flush=True)
                
                # プログレス表示のために小さなバッチで生成
                try:
                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
                    print(" 完了！")
                except torch.cuda.OutOfMemoryError:
                    print("\n⚠️ GPUメモリ不足！CPUフォールバックを試みます...")
                    # GPUメモリをクリアしてCPUで再試行
                    torch.cuda.empty_cache()
                    self.model = self.model.to("cpu")
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    outputs = self.model.generate(**inputs, **generation_config)
            
            elapsed_time = time.time() - start_time
            print(f"✅ SQL生成完了 (処理時間: {elapsed_time:.1f}秒)")
            
            # GPUメモリ状況を表示
            if self.device == "cuda":
                print(f"   最終GPUメモリ使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                torch.cuda.empty_cache()  # メモリをクリア
            
            # デコード
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # プロンプト部分を削除して、生成された部分のみを取得
            sql_query = response[len(prompt):].strip()
            
            # SQLクエリのクリーニング
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            # 最初のSELECT文を抽出
            lines = sql_query.split('\n')
            sql_lines = []
            in_sql = False
            for line in lines:
                if 'SELECT' in line.upper() or in_sql:
                    in_sql = True
                    sql_lines.append(line)
                    if ';' in line:
                        break
            
            sql_query = '\n'.join(sql_lines).strip()
            
            if debug:
                debug_info = {
                    "prompt": prompt,
                    "raw_response": response,
                    "cleaned_sql": sql_query,
                    "model": self.model_name,
                    "device": self.device,
                    "tokens_used": len(inputs['input_ids'][0]) + len(outputs[0])
                }
                return sql_query, debug_info
            
            return sql_query
            
        except Exception as e:
            raise Exception(f"SQL生成エラー: {str(e)}")
    
    def suggest_visualization(self, query, dataframe):
        columns = list(dataframe.columns)
        dtypes = dataframe.dtypes.to_dict()
        sample_data = dataframe.head(5).to_dict('records')
        
        prompt = f"""
以下のクエリ結果に対して、最適な可視化方法を提案してください。

元のクエリ: {query}
カラム: {columns}
データ型: {json.dumps({k: str(v) for k, v in dtypes.items()}, ensure_ascii=False)}
サンプルデータ: {json.dumps(sample_data, ensure_ascii=False, default=str)}

以下から1つだけ選んでください：
- line: 時系列データや連続的な変化
- bar: カテゴリー別の比較
- scatter: 2変数の相関関係
- pie: 構成比の表示
- heatmap: 多次元データの可視化
- table: 表形式での表示

回答は、選択肢の中から1つの単語のみを返してください。
回答: """
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            generation_config = {
                "max_new_tokens": 10,
                "temperature": 0.1,
                "do_sample": True,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            chart_type = response[len(prompt):].strip().lower()
            
            # 最初の単語のみを取得
            chart_type = chart_type.split()[0] if chart_type.split() else "bar"
            
            valid_types = ["line", "bar", "scatter", "pie", "heatmap", "table"]
            
            if chart_type not in valid_types:
                return "bar"
            
            return chart_type
            
        except Exception as e:
            return "bar"