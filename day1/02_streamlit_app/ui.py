import streamlit as st
import pandas as pd
import time
from database import save_to_db, get_chat_history, get_db_count, clear_db
from llm import generate_response
from data import create_sample_evaluation_data
from metrics import get_metrics_descriptions
import random

# カスタムCSS
def load_custom_css():
    """カスタムCSSをロードする"""
    st.markdown("""
    <style>
    /* 全体のスタイル */
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* チャットメッセージのスタイル */
    .user-message {
        background-color: #e6f7ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #1E88E5;
    }
    
    .ai-message {
        background-color: #f1f1f1;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #4CAF50;
    }
    
    /* ヘッダースタイル */
    .section-header {
        padding: 5px 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    /* フィードバックボタンスタイル */
    .feedback-option {
        margin-right: 10px;
        padding: 5px 15px;
        border-radius: 20px;
    }
    
    /* カード要素のスタイル */
    .card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin-bottom: 15px;
        background-color: white;
    }
    
    /* メトリクスカードスタイル */
    .metric-card {
        text-align: center;
        padding: 10px;
        background-color: #f9f9f9;
        border-radius: 5px;
        margin: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# アプリが読み込まれたときにCSSをロード
load_custom_css()

# --- チャットページのUI ---
def display_chat_page(pipe):
    """チャットページのUIを表示する"""
    # セクションヘッダー
    st.markdown('<div class="section-header"><h2>💬 AI チャットアシスタント</h2></div>', unsafe_allow_html=True)
    
    # 入力エリアを改善
    st.markdown("### 質問を入力してください")
    user_question = st.text_area("質問", key="question_input", height=100, 
                                value=st.session_state.get("current_question", ""),
                                placeholder="ここに質問を入力してください...")
    
    # より目立つボタン
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        submit_button = st.button("💬 質問を送信", key="submit_btn", use_container_width=True)
    with col2:
        if "chat_history" in st.session_state and st.session_state.chat_history:
            if st.button("🗑️ リセット", key="reset_btn", use_container_width=True):
                # 会話履歴をリセット
                st.session_state.chat_history = []
                st.session_state.current_question = ""
                st.session_state.current_answer = ""
                st.session_state.response_time = 0.0
                st.session_state.feedback_given = False
                st.rerun()

    # セッション状態の初期化
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = ""
    if "response_time" not in st.session_state:
        st.session_state.response_time = 0.0
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 質問が送信された場合
    if submit_button and user_question:
        st.session_state.current_question = user_question
        st.session_state.current_answer = ""
        st.session_state.feedback_given = False

        with st.spinner("🤔 モデルが回答を生成中..."):
            answer, response_time = generate_response(pipe, user_question)
            st.session_state.current_answer = answer
            st.session_state.response_time = response_time
            
            # チャット履歴に追加
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })
            
            st.rerun()

    # チャット履歴の表示
    st.markdown("### 会話履歴")
    
    if not st.session_state.chat_history:
        st.info("👋 こんにちは！質問を入力して会話を始めましょう。")
    else:
        # チャット履歴を表示
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>👤 あなた:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message"><strong>🤖 AI:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        
        # 最新の回答に対するフィードバック（まだフィードバックされていない場合）
        if not st.session_state.feedback_given and st.session_state.current_answer:
            st.markdown(f'<p style="font-size:0.8em; color:gray;">応答時間: {st.session_state.response_time:.2f}秒</p>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### フィードバック")
            st.write("この回答は役に立ちましたか？")
            display_feedback_form()
        elif st.session_state.feedback_given and st.session_state.current_answer:
            st.success("✅ フィードバックありがとうございます！新しい質問を入力してください。")

def display_feedback_form():
    """フィードバック入力フォームを表示する"""
    with st.form("feedback_form", clear_on_submit=False):
        st.subheader("フィードバック")
        
        # Using styled radio buttons with icons
        feedback_options = ["正確", "部分的に正確", "不正確"]
        feedback_icons = ["✅", "⚠️", "❌"]
        
        # Create radio options with icons
        radio_options = [f"{icon} {option}" for icon, option in zip(feedback_icons, feedback_options)]
        
        # Custom CSS to make radio buttons more visually appealing
        st.markdown("""
        <style>
        div.row-widget.stRadio > div {
            flex-direction: row;
            justify-content: center;
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label {
            background-color: #f0f2f6;
            padding: 10px 15px;
            border-radius: 5px;
            margin-right: 10px;
            text-align: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
            background-color: #e0e2e6;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Use radio button for feedback selection
        selected_option = st.radio(
            "回答の評価",
            options=radio_options,
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Convert selected option back to original feedback value
        selected_index = radio_options.index(selected_option)
        feedback = feedback_options[selected_index]
        
        # 2カラムレイアウトでフォーム要素を整理
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 正確な回答を提案")
            correct_answer = st.text_area("より正確な回答（任意）", 
                                        key="correct_answer_input", 
                                        height=100,
                                        placeholder="AIの回答より正確だと思う回答を入力してください...")
        
        with col2:
            st.markdown("#### コメント")
            feedback_comment = st.text_area("コメント（任意）", 
                                          key="feedback_comment_input", 
                                          height=100,
                                          placeholder="回答に対する感想やアドバイスを入力してください...")
        
        # 送信ボタンをより目立たせる
        submitted = st.form_submit_button("📝 フィードバックを送信", use_container_width=True)
        
        if submitted:
            # フィードバックをデータベースに保存
            is_correct = 1.0 if feedback == "正確" else (0.5 if feedback == "部分的に正確" else 0.0)
            
            # コメントがない場合でも評価を含める
            combined_feedback = f"{feedback}"
            if feedback_comment:
                combined_feedback += f": {feedback_comment}"

            save_to_db(
                st.session_state.current_question,
                st.session_state.current_answer,
                combined_feedback,
                correct_answer,
                is_correct,
                st.session_state.response_time
            )
            st.session_state.feedback_given = True
            st.success("✨ フィードバックが保存されました！新しい質問を入力できます。")
            st.rerun()

# --- 履歴閲覧ページのUI ---
def display_history_page():
    """履歴閲覧ページのUIを表示する"""
    st.markdown('<div class="section-header"><h2>📚 チャット履歴と評価指標</h2></div>', unsafe_allow_html=True)
    
    history_df = get_chat_history()

    if history_df.empty:
        st.info("📭 まだチャット履歴がありません。チャットページで会話を始めましょう。")
        return

    # タブでセクションを分ける
    tab1, tab2 = st.tabs(["📋 履歴閲覧", "📊 評価指標分析"])

    with tab1:
        display_history_list(history_df)

    with tab2:
        display_metrics_analysis(history_df)

def display_history_list(history_df):
    """履歴リストを表示する"""
    st.markdown("### 履歴リスト")
    
    # 検索機能
    search_query = st.text_input("🔍 質問やキーワードで検索", placeholder="検索キーワードを入力...")
    
    # 表示オプション
    filter_options = {
        "すべて表示": None,
        "✅ 正確なもののみ": 1.0,
        "⚠️ 部分的に正確なもののみ": 0.5,
        "❌ 不正確なもののみ": 0.0
    }
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_option = st.radio(
            "表示フィルタ",
            options=filter_options.keys(),
            horizontal=True,
            label_visibility="collapsed"
        )
    
    with col2:
        sort_by = st.selectbox("並び順:", ["新しい順", "古い順", "正確性高い順", "単語数多い順"], index=0)
    
    # フィルタリング
    filter_value = filter_options[display_option]
    
    # 検索フィルタリング
    if search_query:
        filtered_df = history_df[history_df["question"].str.contains(search_query, case=False, na=False) | 
                                history_df["answer"].str.contains(search_query, case=False, na=False)]
    else:
        filtered_df = history_df.copy()
    
    # 正確性フィルタリング
    if filter_value is not None:
        filtered_df = filtered_df[filtered_df["is_correct"].notna() & (filtered_df["is_correct"] == filter_value)]
    
    # ソート
    if sort_by == "新しい順":
        filtered_df = filtered_df.sort_values("timestamp", ascending=False)
    elif sort_by == "古い順":
        filtered_df = filtered_df.sort_values("timestamp", ascending=True)
    elif sort_by == "正確性高い順":
        filtered_df = filtered_df.sort_values("is_correct", ascending=False)
    elif sort_by == "単語数多い順":
        filtered_df = filtered_df.sort_values("word_count", ascending=False)

    if filtered_df.empty:
        st.info("条件に一致する履歴はありません。検索条件やフィルタを変更してみてください。")
        return

    # ページネーション
    items_per_page = 5
    total_items = len(filtered_df)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    col1, col2 = st.columns([5, 1])
    with col2:
        current_page = st.number_input('ページ', min_value=1, max_value=total_pages, value=1, step=1)
    
    with col1:
        st.caption(f"全 {total_items} 件中 {(current_page-1)*items_per_page+1} - {min(current_page*items_per_page, total_items)} 件を表示")

    start_idx = (current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    paginated_df = filtered_df.iloc[start_idx:end_idx]

    # 履歴表示をカード形式に改善
    for i, row in paginated_df.iterrows():
        # 正確性に基づいたアイコンを設定
        if row['is_correct'] == 1.0:
            accuracy_icon = "✅"
            accuracy_color = "green"
        elif row['is_correct'] == 0.5:
            accuracy_icon = "⚠️"
            accuracy_color = "orange" 
        elif row['is_correct'] == 0.0:
            accuracy_icon = "❌"
            accuracy_color = "red"
        else:
            accuracy_icon = "❓"
            accuracy_color = "gray"
        
        expander_title = f"{accuracy_icon} {row['timestamp']} - Q: {row['question'][:50] if row['question'] else 'N/A'}..."
        
        with st.expander(expander_title):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # 質問と回答
            st.markdown(f"<div class='user-message'><strong>👤 ユーザーの質問:</strong><br>{row['question']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='ai-message'><strong>🤖 AIの回答:</strong><br>{row['answer']}</div>", unsafe_allow_html=True)
            
            # フィードバックと正解
            st.markdown(f"<strong>📝 フィードバック:</strong> {row['feedback']}", unsafe_allow_html=True)
            
            if row['correct_answer']:
                st.markdown(f"<strong>✨ 正解例:</strong> {row['correct_answer']}", unsafe_allow_html=True)
            
            # 評価指標の表示を改善
            st.markdown("---")
            st.markdown("<h4>評価指標</h4>", unsafe_allow_html=True)
            
            # 2行3列のグリッドで評価指標を表示
            col1, col2, col3 = st.columns(3)
            
            # 1行目
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color:{accuracy_color};">{row['is_correct']:.1f}</h3>
                    <p>正確性スコア</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{row['response_time']:.2f}秒</h3>
                    <p>応答時間</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{row['word_count']}</h3>
                    <p>単語数</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 2行目
            col1, col2, col3 = st.columns(3)
            with col1:
                bleu_value = f"{row['bleu_score']:.4f}" if pd.notna(row['bleu_score']) else "-"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{bleu_value}</h3>
                    <p>BLEU</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                sim_value = f"{row['similarity_score']:.4f}" if pd.notna(row['similarity_score']) else "-"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{sim_value}</h3>
                    <p>類似度</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                rel_value = f"{row['relevance_score']:.4f}" if pd.notna(row['relevance_score']) else "-"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{rel_value}</h3>
                    <p>関連性</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

def display_metrics_analysis(history_df):
    """評価指標の分析結果を表示する"""
    st.markdown("### 評価指標の分析")
    
    # is_correct が NaN のレコードを除外して分析
    analysis_df = history_df.dropna(subset=['is_correct'])
    if analysis_df.empty:
        st.warning("⚠️ 分析可能な評価データがありません。")
        return

    accuracy_labels = {1.0: '✅ 正確', 0.5: '⚠️ 部分的に正確', 0.0: '❌ 不正確'}
    analysis_df['正確性'] = analysis_df['is_correct'].map(accuracy_labels)
    
    # ダッシュボードレイアウト
    col1, col2 = st.columns(2)
    
    # 正確性の分布
    with col1:
        st.markdown("#### 正確性の分布")
        accuracy_counts = analysis_df['正確性'].value_counts()
        if not accuracy_counts.empty:
            st.bar_chart(accuracy_counts)
        else:
            st.info("正確性データがありません。")
    
    # 応答時間の分布
    with col2:
        st.markdown("#### 応答時間の分布")
        if 'response_time' in analysis_df.columns:
            response_time_data = pd.DataFrame({
                'response_time': analysis_df['response_time']
            })
            st.line_chart(response_time_data.sort_values('response_time'))
        else:
            st.info("応答時間データがありません。")
    
    # 応答時間と他の指標の関係
    st.markdown("#### 応答時間とその他の指標の関係")
    metric_options = ["bleu_score", "similarity_score", "relevance_score", "word_count"]
    metric_names = {"bleu_score": "BLEU スコア", "similarity_score": "類似度", 
                   "relevance_score": "関連性", "word_count": "単語数"}
    
    # 利用可能な指標のみ選択肢に含める
    valid_metric_options = [m for m in metric_options if m in analysis_df.columns and analysis_df[m].notna().any()]

    if valid_metric_options:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            metric_option = st.selectbox(
                "指標を選択:",
                valid_metric_options,
                format_func=lambda x: metric_names.get(x, x),
                key="metric_select"
            )
        
        with col1:
            chart_data = analysis_df[['response_time', metric_option, '正確性']].dropna()
            if not chart_data.empty:
                st.scatter_chart(
                    chart_data,
                    x='response_time',
                    y=metric_option,
                    color='正確性',
                    size=100  # サイズを固定して見やすく
                )
            else:
                st.info(f"選択された指標 ({metric_names.get(metric_option, metric_option)}) と応答時間の有効なデータがありません。")
    else:
        st.info("応答時間と比較できる指標データがありません。")

    # 評価指標の統計情報
    st.markdown("#### 評価指標の統計情報")
    stats_cols = ['response_time', 'bleu_score', 'similarity_score', 'word_count', 'relevance_score']
    valid_stats_cols = [c for c in stats_cols if c in analysis_df.columns and analysis_df[c].notna().any()]
    
    if valid_stats_cols:
        # 統計情報のカード表示
        metrics_stats = analysis_df[valid_stats_cols].describe()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 全体統計")
            st.dataframe(metrics_stats.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        with col2:
            # 正確性レベル別の平均スコア
            st.markdown("##### 正確性レベル別の平均スコア")
            if '正確性' in analysis_df.columns:
                try:
                    accuracy_groups = analysis_df.groupby('正確性')[valid_stats_cols].mean()
                    st.dataframe(accuracy_groups.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
                except Exception as e:
                    st.warning(f"正確性別スコアの集計中にエラーが発生しました: {e}")
            else:
                st.info("正確性レベル別の平均スコアを計算できるデータがありません。")
    else:
        st.info("統計情報を計算できる評価指標データがありません。")

    # カスタム評価指標：効率性スコア
    st.markdown("#### 効率性スコア (正確性 / (応答時間 + 0.1))")
    if 'response_time' in analysis_df.columns and analysis_df['response_time'].notna().any():
        # ゼロ除算を避けるために0.1を追加
        analysis_df['efficiency_score'] = analysis_df['is_correct'] / (analysis_df['response_time'].fillna(0) + 0.1)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 効率性スコアのチャート
            top_efficiency = analysis_df.sort_values('efficiency_score', ascending=False).head(10)
            
            if 'id' in top_efficiency.columns and not top_efficiency.empty:
                chart_data = top_efficiency.set_index('id')[['efficiency_score', '正確性']]
                st.bar_chart(chart_data['efficiency_score'])
            else:
                # IDがない場合は別の列をインデックスとして使用
                chart_data = top_efficiency[['efficiency_score', '正確性']]
                st.bar_chart(chart_data['efficiency_score'])
        
        with col2:
            # 効率性スコアの上位エントリを表示
            st.markdown("##### 効率性スコア上位")
            top_entries = top_efficiency[['efficiency_score', 'is_correct', 'response_time']].reset_index(drop=True)
            st.dataframe(top_entries.style.highlight_max('efficiency_score', color='lightgreen'), use_container_width=True)
    else:
        st.info("効率性スコアを計算するための応答時間データがありません。")

# --- サンプルデータ管理ページのUI ---
def display_data_page():
    """サンプルデータ管理ページのUIを表示する"""
    st.markdown('<div class="section-header"><h2>📊 データ管理とメトリクス説明</h2></div>', unsafe_allow_html=True)
    
    # データ管理セクション
    st.markdown("### サンプル評価データの管理")
    
    # 現在のデータ状況
    count = get_db_count()
    
    # プログレスバーでデータ量を視覚化
    st.progress(min(count/30, 1.0))  # 30件を上限としてスケール
    st.markdown(f"<h4 style='text-align: center;'>現在のデータベースには <span style='color:#4CAF50;'>{count} 件</span> のレコードがあります</h4>", unsafe_allow_html=True)
    
    # ボタンの配置を改善
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📥 サンプルデータを追加", key="create_samples", use_container_width=True):
            create_sample_evaluation_data()
            st.rerun()
    
    with col2:
        if st.button("🔄 データベースをリフレッシュ", key="refresh_db", use_container_width=True):
            st.rerun()
    
    with col3:
        # 確認ステップ付きのクリアボタン
        if st.button("🗑️ データベースをクリア", key="clear_db_button", use_container_width=True):
            if clear_db():
                st.rerun()
    
    # 評価指標の説明セクション
    st.markdown("### 評価指標の説明")
    
    # インタラクティブな指標説明
    metrics_info = get_metrics_descriptions()
    
    # タブで指標をグループ化
    accuracy_tab, timing_tab, similarity_tab = st.tabs(["正確性指標", "タイミング指標", "類似度指標"])
    
    with accuracy_tab:
        for metric in ["正確性スコア (is_correct)", "効率性スコア (efficiency_score)", "関連性スコア (relevance_score)"]:
            if metric in metrics_info:
                with st.expander(f"{metric}"):
                    st.markdown(f"<div class='card'>{metrics_info[metric]}</div>", unsafe_allow_html=True)
    
    with timing_tab:
        for metric in ["応答時間 (response_time)"]:
            if metric in metrics_info:
                with st.expander(f"{metric}"):
                    st.markdown(f"<div class='card'>{metrics_info[metric]}</div>", unsafe_allow_html=True)
    
    with similarity_tab:
        for metric in ["BLEU スコア (bleu_score)", "類似度スコア (similarity_score)", "単語数 (word_count)"]:
            if metric in metrics_info:
                with st.expander(f"{metric}"):
                    st.markdown(f"<div class='card'>{metrics_info[metric]}</div>", unsafe_allow_html=True)
    
    # 簡易的な使い方ガイド
    st.markdown("### 💡 使い方ガイド")
    st.info("""
    1. **チャット**: AIに質問して回答を得られます。回答に対してフィードバックを行うことができます。
    2. **履歴閲覧**: 過去の会話履歴を確認し、評価指標を分析できます。
    3. **データ管理**: テスト用のサンプルデータを追加したり、データベースをクリアできます。
    
    それぞれの評価指標は、回答の品質を異なる視点から測定します。正確性、応答時間、単語数などの指標を組み合わせることで、AIの性能を総合的に評価できます。
    """)