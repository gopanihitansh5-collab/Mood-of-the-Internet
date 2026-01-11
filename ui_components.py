import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def render_header():
    """Render main header with styling"""
    st.markdown("""
        <h1 style='text-align: center; color: #1f77b4;'>
            🌍 Mood of the Internet
        </h1>
        <p style='text-align: center; font-size: 1.2em; color: #666;'>
            Advanced NLP Intelligence Platform for Sentiment, Emotion & Narrative Analysis
        </p>
        <hr style='margin: 20px 0;'>
    """, unsafe_allow_html=True)

def render_metrics(mood, volatility, total_count):
    """Render key metrics in columns"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mood_color = "normal" if 30 <= mood <= 70 else ("inverse" if mood < 30 else "off")
        st.metric("🎯 Mood Score", f"{mood}/100", help="Overall sentiment health (0=negative, 100=positive)")
    
    with col2:
        st.metric("📊 Volatility Index", f"{volatility}%", help="Emotional diversity (higher = more scattered emotions)")
    
    with col3:
        st.metric("📝 Texts Analyzed", f"{total_count:,}", help="Total documents processed")

def render_sentiment_gauge(mood):
    """Render mood score as a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = mood,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Mood Score", 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

def render_charts(sentiment_pct, emotion_pct):
    """Render sentiment and emotion distribution charts"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sentiment Distribution")
        
        colors = {'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#95a5a6'}
        sentiment_df = sentiment_pct.reset_index()
        sentiment_df.columns = ['Sentiment', 'Percentage']
        sentiment_df['Color'] = sentiment_df['Sentiment'].map(colors)
        
        fig = px.bar(
            sentiment_df,
            x='Sentiment',
            y='Percentage',
            color='Sentiment',
            color_discrete_map=colors,
            text='Percentage'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎭 Emotion Distribution")
        
        emotion_df = emotion_pct.reset_index()
        emotion_df.columns = ['Emotion', 'Percentage']
        
        fig = px.bar(
            emotion_df,
            x='Emotion',
            y='Percentage',
            color='Percentage',
            color_continuous_scale='Blues',
            text='Percentage'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

def render_emotion_pie(emotion_pct):
    """Render emotion distribution as pie chart"""
    fig = px.pie(
        values=emotion_pct.values,
        names=emotion_pct.index,
        title="Emotion Breakdown",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def render_topic_visualization(topic_info):
    """Render topic modeling results"""
    if topic_info is None or len(topic_info) == 0:
        st.info("No topics detected. Try analyzing more text data.")
        return
    
    st.subheader("🔍 Discovered Topics & Narratives")
    
    # Filter out outlier topic (-1)
    topics_filtered = topic_info[topic_info['Topic'] != -1].copy()
    
    if len(topics_filtered) == 0:
        st.warning("No clear topics found in the data.")
        return
    
    # Topic size chart
    fig = px.bar(
        topics_filtered.head(10),
        x='Count',
        y='Name',
        orientation='h',
        title=f"Top 10 Topics by Document Count",
        labels={'Count': 'Number of Documents', 'Name': 'Topic'},
        color='Count',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def render_topic_emotion_heatmap(correlation_df):
    """Render heatmap of topic-emotion correlation"""
    if correlation_df is None or len(correlation_df) == 0:
        return
    
    st.subheader("🔥 Topic-Emotion Correlation Heatmap")
    
    # Pivot for heatmap
    heatmap_data = correlation_df.pivot(
        index='topic',
        columns='emotion',
        values='percentage'
    ).fillna(0)
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Emotion", y="Topic ID", color="Percentage"),
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def render_narratives(narratives_df):
    """Render narrative analysis table"""
    if narratives_df is None or len(narratives_df) == 0:
        return
    
    st.subheader("📖 Narrative Intelligence Report")
    
    # Style the dataframe
    styled_df = narratives_df.copy()
    
    # Add emoji indicators
    def sentiment_emoji(score):
        if score > 50:
            return "🟢"
        elif score > 0:
            return "🟡"
        elif score > -50:
            return "🟠"
        else:
            return "🔴"
    
    styled_df['status'] = styled_df['sentiment_score'].apply(sentiment_emoji)
    
    # Reorder columns
    display_cols = ['status', 'narrative', 'document_count', 'sentiment_score', 
                   'dominant_emotion', 'positive_pct', 'negative_pct']
    
    st.dataframe(
        styled_df[display_cols].head(10),
        column_config={
            "status": st.column_config.TextColumn("📊", width="small"),
            "narrative": st.column_config.TextColumn("Narrative Theme", width="large"),
            "document_count": st.column_config.NumberColumn("Mentions", format="%d"),
            "sentiment_score": st.column_config.NumberColumn("Sentiment", format="%+.1f"),
            "dominant_emotion": st.column_config.TextColumn("Emotion"),
            "positive_pct": st.column_config.ProgressColumn("Positive %", format="%.1f", max_value=100),
            "negative_pct": st.column_config.ProgressColumn("Negative %", format="%.1f", max_value=100),
        },
        hide_index=True,
        use_container_width=True
    )

def render_insights(insights_list):
    """Render AI-generated insights"""
    st.subheader("💡 Key Insights")
    
    for insight in insights_list:
        st.markdown(f"• {insight}")

def render_export_section(df, narratives_df=None):
    """Render data export section"""
    st.subheader("📤 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📊 Download Full Analysis (CSV)",
            data=df.to_csv(index=False),
            file_name="mood_analysis_full.csv",
            mime="text/csv"
        )
    
    with col2:
        if narratives_df is not None and len(narratives_df) > 0:
            st.download_button(
                label="📖 Download Narratives (CSV)",
                data=narratives_df.to_csv(index=False),
                file_name="narratives_report.csv",
                mime="text/csv"
            )
        else:
            st.button("📖 Download Narratives (CSV)", disabled=True, 
                     help="No narrative data available")
