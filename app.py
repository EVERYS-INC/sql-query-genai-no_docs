import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from src.database import DatabaseConnection
from src.openai_client import OpenAIClient
from src.visualizer import DataVisualizer

load_dotenv()

st.set_page_config(
    page_title="製造業工場データ分析システム",
    page_icon="🏭",
    layout="wide"
)

def main():
    st.title("🏭 製造業工場データ分析システム")
    st.markdown("自然言語で工場の生産データ、機器稼働状況、品質情報を分析します")
    
    # 初期化時に自動的にSQLiteに接続
    if 'db_connection' not in st.session_state:
        try:
            st.session_state.db_connection = DatabaseConnection(
                db_type="sqlite",
                database="database/factory_data.db"
            )
        except Exception as e:
            st.session_state.db_connection = None
            st.error(f"データベース自動接続エラー: {str(e)}")
    
    if 'openai_client' not in st.session_state:
        try:
            st.session_state.openai_client = OpenAIClient()
        except Exception as e:
            st.session_state.openai_client = None
            st.error(f"Azure OpenAI初期化エラー: {str(e)}")
    
    if 'llama_client' not in st.session_state:
        st.session_state.llama_client = None
    
    if 'gemma_client' not in st.session_state:
        st.session_state.gemma_client = None
    
    if 'llama_device_mode' not in st.session_state:
        st.session_state.llama_device_mode = 'auto'  # 'auto', 'gpu', 'cpu'
    
    with st.sidebar:
        st.header("⚙️ システム状態")
        
        # データベース接続状態の表示
        st.subheader("データベース接続")
        if st.session_state.db_connection:
            st.success("✅ SQLiteデータベース接続済み")
            st.info("📁 database/factory_data.db")
        else:
            st.error("❌ データベース未接続")
        
        # AIモデル選択
        st.subheader("🤖 AIモデル選択")
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = 'azure_openai'
        
        st.session_state.selected_model = st.radio(
            "使用するAIモデル",
            options=['azure_openai', 'llama_elyza', 'gemma_2b'],
            format_func=lambda x: {
                'azure_openai': "☁️ Azure OpenAI (高速・クラウド)",
                'llama_elyza': "🦙 Llama-3-ELYZA-JP-8B (16GB)",
                'gemma_2b': "💎 Gemma-2-2b-jpn-it (軽量・高速)"
            }[x],
            key="model_selector",
            help="Azure OpenAI: クラウド接続必須 | Llama-3: 高精度・16GB | Gemma-2: 軽量・4GB VRAM対応"
        )
        
        # モデル別の接続状態表示
        if st.session_state.selected_model == 'azure_openai':
            if st.session_state.openai_client:
                st.success("✅ Azure OpenAI接続済み")
            else:
                st.error("❌ Azure OpenAI未接続")
        elif st.session_state.selected_model == 'gemma_2b':
            if st.session_state.gemma_client:
                st.success("✅ Gemma-2-2b-jpn-it読み込み済み")
                current_device = "GPU" if hasattr(st.session_state.gemma_client, 'device') and st.session_state.gemma_client.device == "cuda" else "CPU"
                st.info(f"💎 現在のモード: {current_device} (軽量モデル)")
                if st.button("モデルを再読み込み"):
                    st.session_state.gemma_client = None
                    st.rerun()
            else:
                # GPU利用可能性をチェック
                import torch
                gpu_available = torch.cuda.is_available()
                
                if gpu_available:
                    st.info(f"⚡ GPUが利用可能: {torch.cuda.get_device_name(0)}")
                    st.success("🎉 Gemma-2は4GB VRAMでも快適に動作！")
                else:
                    st.info("💻 CPUモードで動作します（Gemmaは軽量なのでCPUでも高速）")
                
                if st.button("モデルを読み込む", type="primary"):
                    with st.spinner("モデルを読み込み中..."):
                        try:
                            from src.gemma_client import GemmaClient
                            # 4GB VRAMの場合は強制的にCPUモードを使用
                            force_cpu = not gpu_available or (gpu_available and torch.cuda.get_device_properties(0).total_memory < 5 * 1024**3)
                            st.session_state.gemma_client = GemmaClient(force_cpu=force_cpu)
                            st.success("✅ モデル読み込み完了")
                            st.rerun()
                        except Exception as e:
                            st.error(f"モデル読み込みエラー: {str(e)}")
                else:
                    st.warning("⚠️ Gemmaモデル未読み込み")
                    st.info("上のボタンをクリックしてモデルを読み込んでください")
        else:  # llama_elyza
            if st.session_state.llama_client:
                st.success("✅ Llama-3-ELYZA-JP-8B読み込み済み")
                current_device = "GPU" if hasattr(st.session_state.llama_client, 'device') and st.session_state.llama_client.device == "cuda" else "CPU"
                st.info(f"💻 現在のモード: {current_device}")
                if st.button("モデルを再読み込み"):
                    st.session_state.llama_client = None
                    st.rerun()
            else:
                # GPU利用可能性をチェック
                import torch
                gpu_available = torch.cuda.is_available()
                
                if gpu_available:
                    st.info(f"⚡ GPUが利用可能です: {torch.cuda.get_device_name(0)}")
                else:
                    st.info("💻 GPUが利用できません。CPUモードで動作します")
                
                # デバイス選択
                device_mode = st.radio(
                    "動作モードを選択",
                    options=['auto', 'gpu', 'cpu'],
                    format_func=lambda x: {
                        'auto': '🅰️ 自動選択 (GPU優先)',
                        'gpu': '⚡ GPUモード (高速)',
                        'cpu': '💻 CPUモード'
                    }[x],
                    key="device_selector",
                    help="GPUモードは高速ですが、NVIDIA GPUが必要です"
                )
                
                if st.button("モデルを読み込む", type="primary"):
                    with st.spinner("モデルを読み込み中..."):
                        try:
                            # デバイスモードに応じてモデルを初期化
                            if device_mode == 'cpu':
                                from src.llama_client_cpu import LlamaClientCPU
                                st.session_state.llama_client = LlamaClientCPU()
                            elif device_mode == 'gpu':
                                if not gpu_available:
                                    st.error("⚠️ GPUが利用できません。CPUモードで読み込みます。")
                                    from src.llama_client_cpu import LlamaClientCPU
                                    st.session_state.llama_client = LlamaClientCPU()
                                else:
                                    try:
                                        from src.llama_client import LlamaClient
                                        st.session_state.llama_client = LlamaClient(force_cpu=False)
                                    except Exception as gpu_error:
                                        st.warning(f"GPU読み込みエラー: {str(gpu_error)}")
                                        st.info("CPUモードにフォールバックします...")
                                        from src.llama_client_cpu import LlamaClientCPU
                                        st.session_state.llama_client = LlamaClientCPU()
                            else:  # auto
                                try:
                                    from src.llama_client import LlamaClient
                                    st.session_state.llama_client = LlamaClient(force_cpu=False)
                                except Exception:
                                    st.info("CPUモードで読み込みます...")
                                    from src.llama_client_cpu import LlamaClientCPU
                                    st.session_state.llama_client = LlamaClientCPU()
                            
                            st.session_state.llama_device_mode = device_mode
                            st.success("✅ モデル読み込み完了")
                            st.rerun()
                        except Exception as e:
                            st.error(f"モデル読み込みエラー: {str(e)}")
                else:
                    st.warning("⚠️ Llamaモデル未読み込み")
                    st.info("上のボタンをクリックしてモデルを読み込んでください")
        
        # デバッグモード
        st.subheader("🔧 デバッグ設定")
        if 'debug_mode' not in st.session_state:
            st.session_state.debug_mode = False
        st.session_state.debug_mode = st.checkbox("デバッグモード", value=st.session_state.debug_mode, key="debug_sidebar")
        if st.session_state.debug_mode:
            st.info("プロンプトとAI出力を表示します")
    
    with st.expander("💡 クエリサンプルを見る"):
        st.markdown("""
        **生産実績分析**
        - 今月の製品別の生産数を教えて
        - 各ラインの今月の生産効率を比較して
        - 不良率が最も高い製品トップ5
        
        **機器稼働分析**
        - 全機器の今日の稼働率を見せて
        - 今週の機器停止時間が長い順に表示
        - 各ラインの今月のOEEを計算して
        
        **品質管理**
        - 今週の品質検査合格率を製品別に表示
        - 不良タイプ別の発生件数を集計
        
        **メンテナンス**
        - 今月実施したメンテナンス一覧
        - メンテナンスコストが高い機器トップ5
        """)
    
    query_input = st.text_area(
        "データベースへの問い合わせを自然言語で入力してください",
        placeholder="例: 今月の各ラインの稼働率を比較して表示",
        height=100
    )
    
    # 選択されたモデルの可用性をチェック
    model_available = False
    if st.session_state.selected_model == 'azure_openai':
        model_available = st.session_state.openai_client is not None
    elif st.session_state.selected_model == 'gemma_2b':
        model_available = st.session_state.gemma_client is not None
    else:  # llama_elyza
        model_available = st.session_state.llama_client is not None
    
    if st.button("クエリ実行", type="primary", disabled=not (st.session_state.db_connection and model_available)):
        if query_input:
            with st.spinner("SQLクエリを生成中..."):
                try:
                    table_info = st.session_state.db_connection.get_table_schema()
                    
                    # セッション状態からデバッグモードを取得
                    debug_mode = st.session_state.get('debug_mode', False)
                    
                    # 選択されたモデルに応じてSQL生成
                    if st.session_state.selected_model == 'azure_openai':
                        ai_client = st.session_state.openai_client
                        model_name = "Azure OpenAI"
                    elif st.session_state.selected_model == 'gemma_2b':
                        ai_client = st.session_state.gemma_client
                        model_name = "Gemma-2-2b-jpn-it"
                    else:  # llama_elyza
                        ai_client = st.session_state.llama_client
                        model_name = "Llama-3-ELYZA-JP-8B"
                    
                    result = ai_client.generate_sql(
                        query_input, 
                        table_info,
                        debug=debug_mode
                    )
                    
                    if debug_mode:
                        sql_query, debug_info = result
                        
                        # デバッグ情報の表示
                        with st.expander("🔍 デバッグ情報", expanded=True):
                            st.subheader(f"{model_name}へのプロンプト")
                            st.text_area("送信プロンプト", debug_info["prompt"], height=300)
                            
                            st.subheader(f"{model_name}からの生のレスポンス")
                            st.code(debug_info["raw_response"], language="text")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("使用モデル", debug_info["model"])
                            with col2:
                                if debug_info["tokens_used"]:
                                    st.metric("使用トークン数", debug_info["tokens_used"])
                    else:
                        sql_query = result
                    
                    # SQLクエリを全幅で表示
                    st.subheader("生成されたSQLクエリ")
                    st.code(sql_query, language="sql")
                    
                    with st.spinner("クエリを実行中..."):
                        df = st.session_state.db_connection.execute_query(sql_query)
                        
                        if df is not None and not df.empty:
                            # クエリ結果を全幅で表示
                            st.subheader("クエリ結果")
                            st.dataframe(df, use_container_width=True)
                            
                            visualizer = DataVisualizer()
                            chart_type = ai_client.suggest_visualization(
                                query_input, df
                            )
                            
                            # グラフを全幅で表示
                            st.subheader("データ可視化")
                            fig = visualizer.create_chart(df, chart_type, query_input)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("クエリ結果が空です")
                            
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
        else:
            st.warning("問い合わせ内容を入力してください")
    
    if st.session_state.db_connection:
        with st.expander("データベーステーブル情報"):
            try:
                schema_info = st.session_state.db_connection.get_table_schema()
                st.code(schema_info, language="text")
            except Exception as e:
                st.error(f"スキーマ取得エラー: {str(e)}")

if __name__ == "__main__":
    main()