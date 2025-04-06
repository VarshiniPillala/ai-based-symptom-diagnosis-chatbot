from flask import Flask, render_template, request, jsonify, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import spacy
import numpy as np
import re

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load dataset
try:
    df = pd.read_csv('symptom2disease.csv')
    df['text'] = df['text'].str.lower()
    df['label'] =  df['label'].str.lower()
except FileNotFoundError:
    print("Error: symptom2disease.csv not found. Download from Kaggle.")
    exit()

# Initialize models
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
model = LogisticRegression(max_iter=1000).fit(X, df['label'])
nlp = spacy.load('en_core_web_sm')

# General question patterns
GENERAL_QUESTIONS = {
    r'what(\'s| is) my (problem|diagnosis|condition)\??': 'diagnosis_request',
    r'what do you think\??': 'diagnosis_request',
    r'what\'?s wrong (with me|with my health)\??': 'diagnosis_request',
    r'do you know (what I have|my problem)\??': 'diagnosis_request',
    r'reset|start over|new diagnosis': 'reset_request'
}

def get_top_predictions(input_vec, threshold=0.1):
    probabilities = model.predict_proba(input_vec)[0]
    top_indices = np.where(probabilities > threshold)[0]
    return [(model.classes_[i], probabilities[i]) for i in 
            sorted(top_indices, key=lambda i: probabilities[i], reverse=True)]

def is_general_question(text):
    """Check if input matches general question patterns"""
    text = text.lower().strip()
    for pattern, q_type in GENERAL_QUESTIONS.items():
        if re.search(pattern, text):
            return q_type
    return False

@app.route('/')
def home():
    session.clear()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').lower().strip()
    
    # Check for general questions first
    question_type = is_general_question(user_input)
    
    if question_type == 'reset_request':
        session.clear()
        return jsonify({
            'type': 'reset',
            'response': "Conversation reset. Please describe your symptoms."
        })
    
    if question_type == 'diagnosis_request':
        if 'symptoms' not in session or not session['symptoms']:
            return jsonify({
                'type': 'general',
                'response': "Please describe your symptoms first (e.g., 'I have headache and fever')."
            })
        else:
            combined_input = ' '.join(session['symptoms'])
            input_vec = vectorizer.transform([combined_input])
            predictions = get_top_predictions(input_vec)
            
            if predictions:
                return jsonify({
                    'type': 'diagnosis',
                    'response': f"Based on your symptoms: {', '.join(session['symptoms'])}\nMost likely: {predictions[0][0]} (confidence: {predictions[0][1]:.0%})",
                    'confidence': float(predictions[0][1]),
                    'condition': predictions[0][0]
                })
    
    # Process as symptom input
    if 'symptoms' not in session:
        session['symptoms'] = []
    
    # Extract symptoms
    doc = nlp(user_input)
    new_symptoms = [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ'] 
                   and token.text not in session['symptoms']]
    session['symptoms'].extend(new_symptoms)
    
    if not session['symptoms']:
        return jsonify({
            'type': 'general',
            'response': "Could you describe your symptoms? Example: 'headache, fever, cough'"
        })
    
    # Make prediction
    combined_input = ' '.join(session['symptoms'])
    input_vec = vectorizer.transform([combined_input])
    predictions = get_top_predictions(input_vec)
    
    if predictions and predictions[0][1] > 0.85:  # Confidence threshold
        diagnosis = predictions[0][0]
        confidence = predictions[0][1]
        session.clear()  # Reset after final diagnosis
        return jsonify({
            'type': 'final_diagnosis',
            'response': f"Final diagnosis: {diagnosis} (confidence: {confidence:.0%})",
            'condition': diagnosis,
            'confidence': float(confidence)
        })
    else:
        return jsonify({
            'type': 'intermediate',
            'response': "Possible conditions:\n" + "\n".join(
                [f"- {pred} ({prob:.0%} confidence)" for pred, prob in predictions[:3]]
            ) + f"\n\nAdditional symptoms needed for accurate diagnosis. (Current: {', '.join(session['symptoms'])})",
            'suggested_questions': [
                "Have you experienced any fever?",
                "How long have you had these symptoms?",
                "Any other discomforts?"
            ]
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='127.0.0.1')