import pandas as pd
import numpy as np
from collections import Counter

def compute_distributions(df):
    """Calculate sentiment and emotion distributions"""
    sentiment_pct = df["sentiment"].value_counts(normalize=True) * 100
    emotion_pct = df["emotion"].value_counts(normalize=True) * 100
    return sentiment_pct, emotion_pct

def mood_score(sentiment_pct, emotion_pct):
    """Calculate overall mood score (0-100)"""
    positive = sentiment_pct.get("POSITIVE", 0)
    negative = sentiment_pct.get("NEGATIVE", 0)
    
    # Negative emotions
    neg_emotions = ["anger", "disgust", "fear", "sadness"]
    emotion_penalty = sum(emotion_pct.get(e, 0) for e in neg_emotions) / 4
    
    # Calculate base score
    base_score = positive - negative
    adjusted_score = base_score - (emotion_penalty * 0.3)
    
    # Scale to 0-100
    final_score = max(0, min(100, 50 + adjusted_score / 2))
    
    return round(final_score, 1)

def volatility_index(emotion_pct):
    """Calculate emotional volatility (how scattered emotions are)"""
    if emotion_pct.empty:
        return 0
    
    # Higher = more concentrated emotions, lower = more scattered
    concentration = (emotion_pct.max() - emotion_pct.min()) / 100
    volatility = 100 * (1 - concentration)
    
    return round(volatility, 1)

def topic_emotion_correlation(df):
    """Find which topics correlate with which emotions"""
    if "topic" not in df.columns:
        return None
    
    correlation_data = []
    
    for topic_id in df["topic"].unique():
        if topic_id == -1:  # Skip outliers
            continue
            
        topic_df = df[df["topic"] == topic_id]
        emotion_dist = topic_df["emotion"].value_counts(normalize=True) * 100
        
        for emotion, pct in emotion_dist.items():
            correlation_data.append({
                "topic": topic_id,
                "emotion": emotion,
                "percentage": round(pct, 1)
            })
    
    return pd.DataFrame(correlation_data)

def detect_narratives(df, topic_info):
    """Identify dominant narratives and their sentiment"""
    if "topic" not in df.columns or topic_info is None:
        return None
    
    narratives = []
    
    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:  # Skip outliers
            continue
        
        topic_docs = df[df["topic"] == topic_id]
        if len(topic_docs) == 0:
            continue
        
        # Get topic name (top words)
        topic_name = row.get("Name", f"Topic {topic_id}")
        
        # Calculate sentiment for this topic
        pos_count = (topic_docs["sentiment"] == "POSITIVE").sum()
        neg_count = (topic_docs["sentiment"] == "NEGATIVE").sum()
        total = len(topic_docs)
        
        sentiment_ratio = (pos_count - neg_count) / total if total > 0 else 0
        
        # Get dominant emotion
        dominant_emotion = topic_docs["emotion"].mode()[0] if len(topic_docs) > 0 else "neutral"
        
        narratives.append({
            "topic_id": topic_id,
            "narrative": topic_name,
            "document_count": total,
            "sentiment_score": round(sentiment_ratio * 100, 1),
            "dominant_emotion": dominant_emotion,
            "positive_pct": round((pos_count / total) * 100, 1) if total > 0 else 0,
            "negative_pct": round((neg_count / total) * 100, 1) if total > 0 else 0
        })
    
    return pd.DataFrame(narratives).sort_values("document_count", ascending=False)

def get_insights(df, sentiment_pct, emotion_pct, mood, narratives=None):
    """Generate AI-style insights from the analysis"""
    insights = []
    
    # Mood insight
    if mood >= 70:
        insights.append(f"🟢 **Highly Positive Mood**: Overall sentiment is very optimistic ({mood}/100)")
    elif mood >= 50:
        insights.append(f"🟡 **Moderately Positive**: Sentiment leans positive but with some concerns ({mood}/100)")
    elif mood >= 30:
        insights.append(f"🟠 **Mixed Sentiment**: Balanced between positive and negative signals ({mood}/100)")
    else:
        insights.append(f"🔴 **Negative Mood**: Predominately negative sentiment detected ({mood}/100)")
    
    # Emotion insight
    dominant_emotion = emotion_pct.idxmax() if not emotion_pct.empty else "neutral"
    emotion_strength = emotion_pct.max() if not emotion_pct.empty else 0
    
    if emotion_strength > 40:
        insights.append(f"**Strong Emotional Signal**: {dominant_emotion.title()} dominates ({emotion_strength:.1f}% of responses)")
    else:
        insights.append(f"**Diverse Emotions**: No single emotion dominates, indicating varied reactions")
    
    # Volume insight
    total_analyzed = len(df)
    insights.append(f"**Sample Size**: {total_analyzed:,} texts analyzed")
    
    # Topic insight
    if narratives is not None and len(narratives) > 0:
        top_narrative = narratives.iloc[0]
        insights.append(f"**Top Narrative**: {top_narrative['narrative']} ({top_narrative['document_count']} mentions, {top_narrative['sentiment_score']:+.0f}% sentiment)")
    
    return insights
