from flask import Flask, render_template, redirect, url_for, request
import pickle
import re
import string
import html
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words('english')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def prediction():  
    text_message = request.form.get('message')
    text = text_message.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\@\w+|\@', '', text)

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords]
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in tokens]

    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
    text = ' '.join(lemma_words)
    
    countvector=TfidfVectorizer(sublinear_tf=True)
    X=countvector.fit_transform([[text]])

    pred = model.predict(X)

    return render_template('result.html', prediction=pred)

if __name__ == "__main__":
	app.run(debug=True)
