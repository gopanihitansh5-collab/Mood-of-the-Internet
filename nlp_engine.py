from transformers import pipeline
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

class NLPEngine:
    """Advanced NLP Engine with sentiment, emotion, and topic modeling"""
    
    def __init__(self):
        print("🔄 Loading NLP models...")
        
        # Sentiment Analysis
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Emotion Detection
        self.emotion = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        
        # Topic Modeling
        self.topic_model = None
        self.topics_fitted = False
        
        print("✅ Models loaded successfully")

    def get_sentiment(self, text):
        """Get sentiment label and confidence score"""
        try:
            result = self.sentiment(text[:512])[0]
            return result["label"], round(result["score"], 3)
        except:
            return "NEUTRAL", 0.5

    def get_emotion(self, text):
        """Get dominant emotion and all emotion scores"""
        try:
            emotions = self.emotion(text[:512])[0]
            dominant = max(emotions, key=lambda x: x["score"])
            return dominant["label"], {e["label"]: round(e["score"], 3) for e in emotions}
        except:
            return "neutral", {}

    def fit_topics(self, texts, min_topic_size=5):
        """Fit BERTopic model on corpus"""
        if len(texts) < min_topic_size:
            print(f"⚠️ Need at least {min_topic_size} texts for topic modeling")
            return None
        
        print(f"🔍 Extracting topics from {len(texts)} documents...")
        
        # Custom vectorizer for better topic words
        vectorizer = CountVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2)
        )
        
        # Initialize and fit BERTopic
        self.topic_model = BERTopic(
            vectorizer_model=vectorizer,
            min_topic_size=min_topic_size,
            nr_topics="auto",
            calculate_probabilities=True
        )
        
        topics, probs = self.topic_model.fit_transform(texts)
        self.topics_fitted = True
        
        print(f"✅ Discovered {len(set(topics)) - 1} topics")
        return topics, probs

    def get_topic_info(self):
        """Get detailed topic information"""
        if not self.topics_fitted:
            return None
        return self.topic_model.get_topic_info()

    def get_topic_words(self, topic_id, top_n=10):
        """Get top words for a specific topic"""
        if not self.topics_fitted:
            return []
        return self.topic_model.get_topic(topic_id)[:top_n]

    def get_document_topic(self, text):
        """Get topic assignment for a single document"""
        if not self.topics_fitted:
            return -1, 0.0
        
        topic, prob = self.topic_model.transform([text])
        return topic[0], prob[0] if len(prob) > 0 else 0.0
