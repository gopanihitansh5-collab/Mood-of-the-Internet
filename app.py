import streamlit as st
import pandas as pd
import time

from nlp_engine import NLPEngine
from analytics import (
    compute_distributions, 
    mood_score, 
    volatility_index,
    topic_emotion_correlation,
    detect_narratives,
    get_insights
)
from ui_components import (
    render_header,
    render_metrics,
    render_sentiment_gauge,
    render_charts,
    render_emotion_pie,
    render_topic_visualization,
    render_topic_emotion_heatmap,
    render_narratives,
    render_insights,
    render_export_section
)

# Page configuration
st.set_page_config(
    page_title="Mood of the Internet",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'df_results' not in st.session_state:
    st.session_state.df_results = None

@st.cache_resource
def load_engine():
    """Load and cache the NLP engine"""
    return NLPEngine()

# Load engine
engine = load_engine()

# Render header
render_header()

# Sidebar configuration
st.sidebar.header("🎛️ Configuration")

# Input mode selection
input_mode = st.sidebar.radio(
    "📥 Input Method:",
    ["Paste Text", "Upload CSV", "Sample Data"],
    help="Choose how to input your data"
)

# Advanced options
with st.sidebar.expander("⚙️ Advanced Settings"):
    enable_topics = st.checkbox("Enable Topic Modeling", value=True, 
                               help="Extract themes and narratives (requires 10+ texts)")
    min_topic_size = st.slider("Minimum Topic Size", 3, 15, 5,
                              help="Minimum documents per topic")
    show_confidence = st.checkbox("Show Confidence Scores", value=False,
                                 help="Display model confidence levels")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.info(
    """
    This platform analyzes text data to extract:
    - **Sentiment**: Positive/Negative/Neutral
    - **Emotions**: Joy, Anger, Fear, etc.
    - **Topics**: Dominant themes & narratives
    - **Insights**: AI-generated intelligence
    """
)

# Main content area
texts = []
sample_loaded = False

if input_mode == "Paste Text":
    st.markdown("### ✍️ Paste Your Text Data")
    raw_text = st.text_area(
        "Enter text (one entry per line):",
        height=300,
        placeholder="The product is amazing!\nTerrible customer service\nI love this brand..."
    )
    texts = [t.strip() for t in raw_text.split("\n") if t.strip()]
    
    if texts:
        st.success(f"✅ {len(texts)} texts ready for analysis")

elif input_mode == "Upload CSV":
    st.markdown("### 📁 Upload CSV File")
    st.info("CSV must contain a column named 'text' with your data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV with a 'text' column"
    )
    
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            if 'text' not in df_upload.columns:
                st.error("❌ CSV must contain a 'text' column")
            else:
                texts = df_upload['text'].dropna().astype(str).tolist()
                st.success(f"✅ Loaded {len(texts)} texts from CSV")
                
                with st.expander("👀 Preview Data"):
                    st.dataframe(df_upload.head(10))
        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}")

else:  # Sample Data
    st.markdown("### 🎲 Using Sample Dataset")
    
    sample_texts = [
        "I absolutely love this product! Best purchase ever.",
        "Terrible experience, would not recommend to anyone.",
        "The customer service was outstanding and very helpful.",
        "Completely disappointed with the quality.",
        "Amazing features and great value for money!",
        "The shipping was delayed and communication was poor.",
        "Exceeded all my expectations, truly impressive.",
        "Waste of money, nothing works as advertised.",
        "Fantastic design and very easy to use.",
        "Horrible quality control, arrived damaged.",
        "This changed my life! Absolutely incredible.",
        "Misleading marketing and poor product quality.",
        "Best investment I've made this year.",
        "Customer support never responded to my emails.",
        "The interface is intuitive and beautifully designed.",
        "Breaking after just two weeks of use.",
        "Innovative solution to a common problem.",
        "Overpriced and underdelivered on promises.",
        "Seamless integration with existing tools.",
        "Buggy software with frequent crashes.",
        "The new update brought amazing improvements.",
        "Data privacy concerns make me uncomfortable.",
        "Revolutionary approach to the industry.",
        "Feels like a downgrade from previous version.",
        "Exceptional build quality and attention to detail."
    ]
    
    texts = sample_texts
    sample_loaded = True
    st.info(f"📊 Loaded {len(texts)} sample reviews for demonstration")

# Analysis button
st.markdown("---")

if st.button("🚀 Analyze Mood & Narratives", type="primary", disabled=len(texts) == 0):
    if len(texts) == 0:
        st.warning("⚠️ Please provide some text data first")
    else:
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize results dataframe
        df = pd.DataFrame({"text": texts})
        
        # Step 1: Sentiment Analysis
        status_text.text("🔍 Analyzing sentiment...")
        progress_bar.progress(20)
        
        sentiments = []
        sent_scores = []
        for text in texts:
            label, score = engine.get_sentiment(text)
            sentiments.append(label)
            sent_scores.append(score)
        
        df["sentiment"] = sentiments
        if show_confidence:
            df["sentiment_score"] = sent_scores
        
        # Step 2: Emotion Analysis
        status_text.text("🎭 Detecting emotions...")
        progress_bar.progress(40)
        
        emotions = []
        emotion_scores_list = []
        for text in texts:
            emotion, scores = engine.get_emotion(text)
            emotions.append(emotion)
            emotion_scores_list.append(scores)
        
        df["emotion"] = emotions
        
        # Step 3: Topic Modeling
        if enable_topics and len(texts) >= min_topic_size:
            status_text.text("🔍 Extracting topics and narratives...")
            progress_bar.progress(60)
            
            topics, probs = engine.fit_topics(texts, min_topic_size=min_topic_size)
            df["topic"] = topics
            if show_confidence:
                df["topic_probability"] = [p.max() if len(p) > 0 else 0 for p in probs]
            
            topic_info = engine.get_topic_info()
        else:
            topic_info = None
        
        # Step 4: Analytics
        status_text.text("📊 Computing analytics...")
        progress_bar.progress(80)
        
        sentiment_pct, emotion_pct = compute_distributions(df)
        mood = mood_score(sentiment_pct, emotion_pct)
        volatility = volatility_index(emotion_pct)
        
        # Advanced analytics
        if enable_topics and topic_info is not None:
            correlation_df = topic_emotion_correlation(df)
            narratives_df = detect_narratives(df, topic_info)
        else:
            correlation_df = None
            narratives_df = None
        
        # Generate insights
        insights = get_insights(df, sentiment_pct, emotion_pct, mood, narratives_df)
        
        # Complete
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Store results
        st.session_state.analysis_complete = True
        st.session_state.df_results = df
        st.session_state.sentiment_pct = sentiment_pct
        st.session_state.emotion_pct = emotion_pct
        st.session_state.mood = mood
        st.session_state.volatility = volatility
        st.session_state.topic_info = topic_info
        st.session_state.correlation_df = correlation_df
        st.session_state.narratives_df = narratives_df
        st.session_state.insights = insights

# Display results
if st.session_state.analysis_complete:
    st.markdown("---")
    st.markdown("## 📈 Analysis Results")
    
    # Key metrics
    render_metrics(
        st.session_state.mood,
        st.session_state.volatility,
        len(st.session_state.df_results)
    )
    
    st.markdown("---")
    
    # Two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sentiment and emotion charts
        render_charts(
            st.session_state.sentiment_pct,
            st.session_state.emotion_pct
        )
        
        # Topic visualization
        if st.session_state.topic_info is not None:
            st.markdown("---")
            render_topic_visualization(st.session_state.topic_info)
            
            # Topic-emotion heatmap
            if st.session_state.correlation_df is not None:
                st.markdown("---")
                render_topic_emotion_heatmap(st.session_state.correlation_df)
    
    with col2:
        # Mood gauge
        render_sentiment_gauge(st.session_state.mood)
        
        # Emotion pie chart
        render_emotion_pie(st.session_state.emotion_pct)
        
        # Insights
        st.markdown("---")
        render_insights(st.session_state.insights)
    
    # Narratives section
    if st.session_state.narratives_df is not None:
        st.markdown("---")
        render_narratives(st.session_state.narratives_df)
    
    # Data exploration
    st.markdown("---")
    with st.expander("🔍 Explore Raw Data"):
        st.dataframe(
            st.session_state.df_results,
            use_container_width=True,
            height=400
        )
    
    # Export section
    st.markdown("---")
    render_export_section(
        st.session_state.df_results,
        st.session_state.narratives_df
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌍 <b>Mood of the Internet</b> | Advanced NLP Intelligence Platform</p>
        <p style='font-size: 0.9em;'>Powered by Transformers • BERTopic • Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
