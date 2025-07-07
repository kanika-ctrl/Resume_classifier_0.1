import pandas as pd
import numpy as np
import re
import nltk
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Download resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

# Read dataset
df = pd.read_csv(r"C:\Users\91720\OneDrive\Desktop\jupyter porjects\Resume_classifier_0.1\archive (3)\UpdatedResumeDataSet.csv")
df.dropna(inplace=True)

# Text cleaning
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

df['Cleaned'] = df['Resume'].apply(clean_text)

# Encode target
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Category'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['Cleaned'], df['Label'], test_size=0.2, stratify=df['Label'], random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=15000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)

def resume_input(resume_text):
    """Predict resume category from input text"""
    cleaned_input = clean_text(resume_text)
    vector_input = vectorizer.transform([cleaned_input])
    predicted_label = model.predict(vector_input)[0]
    predicted_category = le.inverse_transform([predicted_label])[0]
    return predicted_category 