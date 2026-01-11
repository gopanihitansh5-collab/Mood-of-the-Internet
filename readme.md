# 🌍 Mood of the Internet

**Advanced NLP Intelligence Platform for Sentiment, Emotion & Narrative Analysis**

A production-ready, modular platform that analyzes text data to extract sentiment, emotions, dominant narratives, and actionable insights using state-of-the-art NLP models.

---

## ✨ Features

### Core Capabilities
- **Sentiment Analysis**: Detect positive, negative, and neutral sentiment with confidence scores
- **Emotion Detection**: Identify 7 core emotions (joy, anger, fear, sadness, surprise, disgust, neutral)
- **Topic Modeling**: Automatically discover themes and narratives using BERTopic
- **Mood Scoring**: Quantitative 0-100 mood score with volatility index
- **Narrative Intelligence**: Correlate topics with emotions and sentiment
- **Visual Analytics**: Interactive charts, gauges, heatmaps, and distributions
- **Export Capabilities**: Download results as CSV for further analysis

### Technical Highlights
✅ Modular architecture (separation of concerns)  
✅ Caching & performance optimization  
✅ Multi-input support (text paste, CSV upload, sample data)  
✅ Explainability & insights generation  
✅ Scalable to large datasets  
✅ Production-ready with Docker support  
✅ Optional FastAPI backend  

---

## 🏗️ Project Structure

```
mood_of_internet/
├── app.py                 # Main Streamlit application
├── nlp_engine.py          # NLP models (sentiment, emotion, topics)
├── analytics.py           # Analytics & insights computation
├── ui_components.py       # Reusable UI components
├── api.py                 # FastAPI backend (optional)
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker container configuration
├── docker-compose.yml     # Docker orchestration
├── README.md              # Documentation (you are here)
└── data/                  # Data directory (created on first run)
```

---

## 🚀 Quick Start

### Option 1: Local Installation

```bash
# Clone or download the project
cd mood_of_internet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

### Option 2: Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### Option 3: FastAPI Backend

```bash
# Install additional dependencies
pip install fastapi uvicorn

# Run the API server
uvicorn api:app --reload

# API documentation available at:
# http://localhost:8000/docs
```

---

## 📊 Usage Guide

### 1. **Paste Text**
- Enter text data (one entry per line)
- Great for quick analysis of reviews, comments, tweets

### 2. **Upload CSV**
- CSV must contain a column named `text`
- Supports large datasets (thousands of rows)
- Example CSV structure:
  ```csv
  text
  "This product is amazing!"
  "Terrible customer service"
  "Neutral experience overall"
  ```

### 3. **Sample Data**
- Pre-loaded with 25 sample reviews
- Perfect for testing and demonstration

### Advanced Settings
- **Enable Topic Modeling**: Extract themes and narratives (requires 10+ texts)
- **Minimum Topic Size**: Control topic granularity (3-15)
- **Show Confidence Scores**: Display model confidence levels

---

## 🎯 Key Outputs

### Metrics Dashboard
- **Mood Score** (0-100): Overall sentiment health
- **Volatility Index**: Emotional diversity measure
- **Text Count**: Total documents analyzed

### Visualizations
- Sentiment distribution (positive/negative/neutral)
- Emotion breakdown (7 emotion categories)
- Topic clusters and sizes
- Topic-emotion correlation heatmap
- Mood gauge with thresholds

### Narrative Intelligence
- Top themes and their sentiment scores
- Dominant emotions per topic
- Document counts per narrative
- Positive/negative percentages

### Insights
- AI-generated key takeaways
- Trend identification
- Anomaly detection
- Actionable recommendations

---

## 🔧 Technical Details

### NLP Models
- **Sentiment**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Emotion**: `j-hartmann/emotion-english-distilroberta-base`
- **Topics**: BERTopic with UMAP and HDBSCAN

### Performance
- Caching via `@st.cache_resource` for model loading
- Efficient batch processing
- Optimized for datasets up to 10,000 texts
- Handles 100+ texts in < 30 seconds on standard hardware

### Scalability Considerations
- For datasets > 10,000 texts, consider:
  - Using the FastAPI backend
  - Implementing async processing
  - Adding Redis/Celery for job queuing
  - Database integration for results storage

---

## 📦 API Documentation

### Endpoints

#### Health Check
```http
GET /health
```

#### Single Sentiment Analysis
```http
POST /analyze/sentiment
Content-Type: application/json

{
  "text": "This is amazing!"
}
```

#### Single Emotion Analysis
```http
POST /analyze/emotion
Content-Type: application/json

{
  "text": "I'm so frustrated with this"
}
```

#### Bulk Analysis
```http
POST /analyze/bulk
Content-Type: application/json

{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "enable_topics": true,
  "min_topic_size": 5
}
```

Full API documentation: `http://localhost:8000/docs`

---

## 🐳 Docker Deployment

### Build Custom Image
```bash
docker build -t mood-analyzer:latest .
```

### Run Container
```bash
docker run -p 8501:8501 mood-analyzer:latest
```

### Environment Variables
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

---

## 🔮 Future Enhancements

### Planned Features
- [ ] Time-series mood tracking
- [ ] Platform comparison (Twitter vs Reddit vs Reviews)
- [ ] Real-time API ingestion (Twitter, Reddit)
- [ ] Advanced topic evolution visualization
- [ ] Multi-language support
- [ ] Custom model fine-tuning
- [ ] User authentication
- [ ] Database integration (PostgreSQL)
- [ ] Asynchronous batch processing
- [ ] A/B testing framework
- [ ] Keyword-emotion heatmaps
- [ ] Geolocation-based analysis

### Extensibility
This platform is designed to be extended. Consider adding:
- **Data Sources**: Integrate with social media APIs, RSS feeds, databases
- **Models**: Swap in custom fine-tuned models for domain-specific analysis
- **Analytics**: Add custom scoring algorithms, anomaly detection, forecasting
- **Visualization**: Create domain-specific dashboards and reports
- **Deployment**: Scale to cloud platforms (AWS, GCP, Azure)

---

## 🛠️ Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Code Quality
```bash
# Format code
black .

# Lint
flake8 .

# Type checking
mypy .
```

---

## 📈 Use Cases

### Business Intelligence
- Customer feedback analysis
- Product review monitoring
- Brand sentiment tracking
- Market research

### Social Media
- Campaign performance analysis
- Community sentiment monitoring
- Influencer impact assessment
- Crisis detection

### Research
- Public opinion studies
- Narrative analysis
- Emotion dynamics research
- Discourse analysis

### Content Strategy
- Audience engagement analysis
- Content performance optimization
- Topic trend identification
- Emotional resonance testing

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Hugging Face** for Transformers library
- **BERTopic** by Maarten Grootendorst
- **Streamlit** for the amazing web framework
- **FastAPI** for modern API development

---

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com
- Documentation: [Full docs](https://github.com/yourrepo/docs)

---

## 📊 Project Stats

![Python Version](https://img.shields.io/badge/python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)

---

**Built with ❤️ for the NLP community**

*"Understanding the mood of the internet, one text at a time."*
