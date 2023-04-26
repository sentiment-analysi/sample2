import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.utils import resample
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and preprocessing objects
model = load_model('sentiment_analysis_model.h5')

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('history.pkl', 'rb') as f:
    history = pickle.load(f)

# Function to perform sentiment analysis
def predict_sentiment(input_review):
    # Preprocess input text
    input_text = re.sub(pattern='[^a-zA-Z]',repl=' ', string=input_text)
    input_text = input_text.lower()
    input_words = input_text.split()
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    input_words = [word for word in input_words if not word in stop_words]
    lemmatizer = WordNetLemmatizer()
    input_words = [lemmatizer.lemmatize(word) for word in input_words]
    misspelled_words = spell.unknown(input_words)
    corrected_words = [spell.correction(word) if word in misspelled_words else word for word in input_words if word is not None and spell.correction(word) is not None]
    input_text = ' '.join(corrected_words)

    # Vectorize and scale input text
    input_vector = vectorizer.transform([input_text]).toarray()
    input_vector = scaler.transform(input_vector)

    # Predict sentiment of input text
    prediction = model.predict(input_vector)[0][0]
    sentiment = 'positive' if prediction > 0.5 else 'negative'

    return sentiment

# Main function to run the app
def main():
    st.title('Student sentiment analysis')

    # Get the user inputs
    review1 = st.text_input('How was the course experience?')
    review2 = st.text_input('Tell us about the instructor?')
    review3 = st.text_input('Was the material provided useful?')

    # Perform sentiment analysis and show the results
    if st.button('Predict'):
        result1 = predict_sentiment(review1)
        result2 = predict_sentiment(review2)
        result3 = predict_sentiment(review3)
        st.success(f"Course experience: {result1}")
        st.success(f"Instructor: {result2}")
        st.success(f"Material: {result3}")
        
        # Show analytics using a bar chart
        results = {'Course experience': result1, 'Instructor': result2, 'Useful material': result3}
        df = pd.DataFrame({'Reviews': list(results.keys()), 'Sentiment': list(results.values())})
        df_counts = df['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(df_counts.index, df_counts.values, color=['blue', 'yellow'])
        ax.set_title('Sentiment Analysis Results')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        st.pyplot(fig)

# Run the app
if __name__=='__main__':
    main()
