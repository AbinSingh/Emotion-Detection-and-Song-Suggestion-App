## Import the required libraries ##
from flask_cors import CORS 
import pymongo 
import os
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


# connect to the mongoDB cloud
#connection_url='mongodb+srv://abin:dbpassword@cluster0.b8byd.mongodb.net/test?retryWrites=true&w=majority'
connection_url = os.environ.get('MONGODB_URL')

app = Flask(__name__)
client = pymongo.MongoClient(connection_url)

# Access the Database 
Database = client.get_database('Example') 
# Access the Table 
SampleTable = Database.SampleTable

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/getdelay', methods=['POST'])
def get_delay():
    # get the input from the html page
    result=request.form
    
    twitter_post=result['twitter_post']
    trial=[twitter_post]

    # load the model and vectorizer from disk
    log_model = joblib.load('Log_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
 
    future_val=vectorizer.transform(trial).toarray()
    prediction=log_model.predict(future_val)
    
    # To write back to the database
    pred=str(prediction[0])
    dict_val={'0':'anger','3':'Love','1':'Other','2':'Other','4':'Other','5':'Other'}
    output=dict_val[pred]
    
    mydict = { "name": twitter_post, "address":output}

    SampleTable.insert_one(mydict)
    
    # To produce the prediction in the html page and suggest songs

    if prediction[0]==0:
       return render_template('result22.html')
       
    elif prediction[0]==3:
       return render_template('result1.html')

    else:
       return render_template('home.html')
      

if __name__ == '__main__':
    app.run(debug=True)
