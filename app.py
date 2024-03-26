# app.py
"""
Sentiment Analysis Web App using Flask and Keras

Before running the code, make sure to install the required packages:
- Flask: pip install Flask
- TensorFlow: pip install tensorflow
- NLTK: pip install nltk

Make sure to have the 'sentiment_model.h5' file in the same directory as this script.

To run the application, execute the following command in the terminal:
python app.py

Open your web browser and navigate to http://127.0.0.1:5000/ to access the web app.

Usage:
1. Enter a tweet in the provided form on the webpage.
2. Click the "Analyze" button to get the sentiment analysis result.

Note: This code assumes a pre-trained sentiment analysis model ('sentiment_model.h5') is available.
"""
# app.py
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the pre-trained model
model = load_model('sentiment_model.h5')

# removing stopwords --------------------------------------------
stop_words = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    words = [word for word in words if word not in stop_words]
    clean = ' '.join(words)
    return clean
#---------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        tweet = request.form['tweet']
        cleaned_tweet = clean_text(tweet)
        tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        cleaned_tweet = clean_text(tweet)
        tokenizer.fit_on_texts(cleaned_tweet)
        word_index = tokenizer.word_index
        vocab_size = len(word_index) + 1
        user_sequence = pad_sequences(tokenizer.texts_to_sequences([cleaned_tweet]), maxlen=50, padding='post', truncating='post')
        prediction = model.predict(user_sequence)[0, 0]
        sentiment = 0
        if prediction >0.5:
            sentiment = 'Positive'
        else:
            sentiment = 'negative'
        return render_template('index.html', tweet=tweet, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
