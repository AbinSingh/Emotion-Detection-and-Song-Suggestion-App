
from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer



import numpy as np
import json
import pandas as pd
# load the built-in model 


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/getdelay', methods=['POST'])
def get_delay():
    result=request.form

    twitter_post=result['twitter_post']
    trial=[twitter_post]

    # load the model from disk
    log_model = joblib.load('Log_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
 
    future_val=vectorizer.transform(trial).toarray()
    prediction=log_model.predict(future_val)


    if prediction[0]==0:
       return render_template('result22.html')
       
    elif prediction[0]==3:
       return render_template('result1.html')

    else:
       return render_template('home.html')
      

if __name__ == '__main__':
    app.run(debug=True)