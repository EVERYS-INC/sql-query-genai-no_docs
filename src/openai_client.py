import os
import json
import httpx

class OpenAIClient:
    def __init__(self):
        # 必要な設定を取得
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not all([api_key, endpoint, self.deployment_name]):
            raise ValueError("Azure OpenAI設定が不完全です。.envファイルを確認してください。")
        
        # エンドポイントの末尾のスラッシュを削除
        if endpoint.endswith('/'):
            endpoint = endpoint[:-1]
        
        try:
            # カスタムHTTPクライアントを作成（プロキシなし）
            http_client = httpx.Client(
                trust_env=False,  # 環境変数からプロキシ設定を読み込まない
                timeout=30.0
            )
            
            from openai import AzureOpenAI
            
            # プロキシなしでクライアントを初期化
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
                http_client=http_client
            )
        except Exception as e:
            # フォールバック: 基本的な初期化
            try:
                from openai import AzureOpenAI
                self.client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=endpoint,
                    api_version=api_version
                )
            except Exception as e2:
                raise ValueError(f"Azure OpenAI初期化エラー: {str(e2)}")
    
    def generate_sql(self, natural_language_query, table_schema, debug=False):
        prompt = f"""
あなたは製造業の工場データベースに精通したSQL専門家です。
**重要: このデータベースはSQLiteを使用しています。SQLite固有の構文を使用してください。**

以下のテーブル構造を持つデータベースに対して、ユーザーの自然言語での問い合わせをSQLクエリに変換してください。

このデータベースは製造業の工場データを管理しており、以下の主要な情報を含みます：
- 生産ライン（production_lines）: 工場の生産ライン情報
- 製品（products）: 製造される製品のマスタ情報
- 機器（machines）: 各ラインに設置された機器情報
- 生産実績（production_records）: 日々の生産実績データ
- 機器稼働（machine_operations）: 機器の稼働状況（稼働時間、停止時間等）
- 品質検査（quality_inspections）: 製品の品質検査結果
- メンテナンス（maintenance_records）: 機器のメンテナンス履歴

重要な指標の計算方法：
- 稼働率 = running_minutes / (running_minutes + stop_minutes + idle_minutes) * 100
- 品質率 = good_quantity / actual_quantity * 100
- 生産効率 = actual_quantity / planned_quantity * 100
- OEE（設備総合効率）= 稼働率 × 性能率 × 品質率

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
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "あなたはSQL専門家です。正確で安全なSQLクエリのみを生成してください。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            if debug:
                debug_info = {
                    "prompt": prompt,
                    "raw_response": response.choices[0].message.content,
                    "cleaned_sql": sql_query,
                    "model": self.deployment_name,
                    "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
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
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "データ可視化の専門家として、最適なグラフタイプを1つの単語で答えてください。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            chart_type = response.choices[0].message.content.strip().lower()
            valid_types = ["line", "bar", "scatter", "pie", "heatmap", "table"]
            
            if chart_type not in valid_types:
                return "bar"
            
            return chart_type
            
        except Exception as e:
            return "bar"