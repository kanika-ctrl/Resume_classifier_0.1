import streamlit as st
import pandas as pd
import numpy as np
import re
import PyPDF2
import docx
import io
from pathlib import Path
from resume import resume_input, vectorizer, le, clean_text

# Page config
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 Resume Classification System")
st.markdown("Upload your resume (PDF/DOCX/TXT) and get instant classification with detailed analysis!")

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return None

def extract_text_from_txt(txt_file):
    """Extract text from TXT file"""
    try:
        text = txt_file.read().decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error reading TXT: {e}")
        return None

def get_top_features(text, vectorizer, model, top_n=10):
    """Get top features that contributed to the classification"""
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients for the predicted class
    coefficients = model.coef_[model.predict(vector)[0]]
    
    # Get feature values for this text
    feature_values = vector.toarray()[0]
    
    # Calculate feature importance (coefficient * feature_value)
    feature_importance = coefficients * feature_values
    
    # Get top features
    top_indices = np.argsort(feature_importance)[-top_n:][::-1]
    
    top_features = []
    for idx in top_indices:
        if feature_values[idx] > 0:  # Only include features present in the text
            top_features.append({
                'feature': feature_names[idx],
                'importance': feature_importance[idx],
                'value': feature_values[idx]
            })
    
    return top_features

def get_prediction_confidence(text, vectorizer, model):
    """Get prediction confidence scores"""
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(vector)[0]
    predicted_class = model.predict(vector)[0]
    confidence = probabilities[predicted_class]
    return confidence, probabilities

# Load model
# with st.spinner("Loading the classification model..."):
#     model, vectorizer, le, clean_text = load_model()

# if model is None:
#     st.stop()

# File upload section
st.header("📤 Upload Resume")
uploaded_file = st.file_uploader(
    "Choose a resume file",
    type=['pdf', 'docx', 'txt'],
    help="Supported formats: PDF, DOCX, TXT"
)

# Text input as alternative
st.subheader("Or paste resume text directly:")
resume_text = st.text_area("Paste your resume text here:", height=200)

# Process button
process_button = st.button("🚀 Analyze Resume", type="primary")

if process_button:
    if uploaded_file is not None or resume_text.strip():
        with st.spinner("Processing your resume..."):
            
            # Extract text
            if uploaded_file is not None:
                file_extension = Path(uploaded_file.name).suffix.lower()
                
                if file_extension == '.pdf':
                    extracted_text = extract_text_from_pdf(uploaded_file)
                elif file_extension == '.docx':
                    extracted_text = extract_text_from_docx(uploaded_file)
                elif file_extension == '.txt':
                    extracted_file = io.BytesIO(uploaded_file.read())
                    extracted_text = extract_text_from_txt(extracted_file)
                else:
                    st.error("Unsupported file format")
                    st.stop()
                    
                if extracted_text is None:
                    st.error("Could not extract text from the uploaded file")
                    st.stop()
            else:
                extracted_text = resume_text
            
            # Clean text
            cleaned_text = clean_text(extracted_text)
            
            # Make prediction
            predicted_category = resume_input(extracted_text)
            
            # Get confidence
            confidence, all_probabilities = get_prediction_confidence(extracted_text, vectorizer, resume_input.__globals__['model'])
            
            # Get top features
            top_features = get_top_features(extracted_text, vectorizer, resume_input.__globals__['model'], top_n=10)
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Create columns for results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Classification Results")
                
                # Prediction with confidence
                st.metric(
                    label="Predicted Category",
                    value=predicted_category,
                    delta=f"{confidence:.1%} confidence"
                )
                
                # Confidence gauge
                st.progress(confidence)
                st.caption(f"Confidence: {confidence:.1%}")
                
                # All category probabilities
                st.subheader("📈 Category Probabilities")
                categories = le.classes_
                prob_df = pd.DataFrame({
                    'Category': categories,
                    'Probability': all_probabilities
                }).sort_values('Probability', ascending=False)
                
                st.bar_chart(prob_df.set_index('Category'))
                
            with col2:
                st.subheader("🔍 Top Matching Features")
                
                if top_features:
                    for i, feature in enumerate(top_features):
                        importance = feature['importance']
                        feature_name = feature['feature']
                        
                        # Color coding based on importance
                        if importance > 0.5:
                            color = "🟢"
                        elif importance > 0.2:
                            color = "🟡"
                        else:
                            color = "🔴"
                        
                        st.write(f"{color} **{feature_name}** (importance: {importance:.3f})")
                else:
                    st.info("No significant features found")
            
            # Extracted text section
            # st.subheader("📝 Extracted Text")
            # with st.expander("View extracted text"):
            #     st.text_area("Cleaned Text:", cleaned_text, height=200, disabled=True)
            
            # # Raw text
            # with st.expander("View original text"):
            #     st.text_area("Original Text:", extracted_text, height=200, disabled=True)
            
            # # Model performance info
            # st.subheader("ℹ️ Model Information")
            # col1, col2, col3 = st.columns(3)
            
            # with col1:
            #     st.metric("Total Categories", len(le.classes_))
            # with col2:
            #     st.metric("Available Categories", ", ".join(le.classes_[:3]) + "...")
            # with col3:
            #     st.metric("Model Type", "Logistic Regression")
                
    else:
        st.warning("Please upload a file or paste resume text to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit • Resume Classification System</p>
</div>
""", unsafe_allow_html=True)
