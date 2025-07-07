# Resume Classification System

A machine learning-based resume classification system that automatically categorizes resumes into different job categories using NLP and machine learning techniques.


->> https://app1py-g4mactsmrzfkqtqfdkxaz6.streamlit.app/
## Features

- 📄 **Multi-format Support**: Upload PDF and TXT files
- 🤖 **AI-Powered Classification**: Uses TF-IDF and Logistic Regression
- 📊 **Detailed Analysis**: Shows confidence scores and feature importance
- 🎯 **Multiple Categories**: Classifies into various job categories
- 💻 **Web Interface**: Beautiful Streamlit-based UI

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project**
   ```bash
   cd Resume_classifier_0.1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SpaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Start the application** using the command above
2. **Upload a resume** (PDF or TXT format) or paste text directly
3. **Click "Analyze Resume"** to get instant classification
4. **View results** including:
   - Predicted job category
   - Confidence score
   - Category probabilities
   - Top matching features

## Deployment Options

### Local Deployment
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy automatically

### Heroku Deployment
1. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Add `streamlit` to requirements.txt
3. Deploy to Heroku

### Docker Deployment
1. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   RUN python -m spacy download en_core_web_sm
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```
2. Build and run:
   ```bash
   docker build -t resume-classifier .
   docker run -p 8501:8501 resume-classifier
   ```

## Project Structure

```
Resume_classifier_0.1/
├── app.py                 # Main Streamlit application
├── resume.py             # ML model and preprocessing logic
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── archive (3)/
    └── UpdatedResumeDataSet.csv  # Training dataset
```

## Model Details

- **Algorithm**: Logistic Regression with TF-IDF vectorization
- **Features**: N-gram features (1-2 grams)
- **Preprocessing**: Text cleaning, lemmatization, stop word removal
- **Performance**: Optimized for multi-class classification

## Supported Categories

The model can classify resumes into various job categories including:
- Data Science
- Software Development
- Marketing
- Sales
- Human Resources
- And more...

## Troubleshooting

### Common Issues

1. **SpaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **NLTK data missing**
   - The app automatically downloads required NLTK data

3. **Port already in use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

## Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving the model
- Enhancing the UI

## License

This project is open source and available under the MIT License.
