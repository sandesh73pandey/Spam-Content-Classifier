from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import pickle
nltk.download('punkt')
nltk.download('stopwords')
from Services.pfeature import FeatureExtraction
from sklearn.metrics import accuracy_score,precision_score





app = Flask(__name__)

# Load the trained model and vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
clf_model = pickle.load(open('SVMmodel.pkl', 'rb'))
phishing_model = pickle.load(open('phishing_final.pkl', 'rb'))

ps= PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y= []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)





@app.route('/')
def index():
    return render_template('index.html', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    print("hello")
    try:
        if request.method == 'POST':
            if 'message' not in request.form or not request.form['message'].strip():
                return render_template('index.html', prediction=None, error='Text field is required.')
            
            
            text = request.form['message']
            input_text = text
            
            url=request.form['URL']
            
            phishing_input=FeatureExtraction(url)
            
            
            
            transformed_text = transform_text(input_text)
            X_new = vectorizer.transform([transformed_text]).toarray()
            prediction = clf_model.predict(X_new)
            
            
            
            
            phishing_prediction=phishing_model.predict([phishing_input.features])
            print("Result:",prediction)
            
            print("Phising Result:",phishing_prediction)
            
            
            
            
            # if 1 in prediction:
            #     return render_template('spam.html')
            # else:
            #     return render_template('ham.html')
            
            
            # if 1 in prediction and phishing_prediction == -1:
            #     return render_template('spamandphishing.html')
            # elif 0 in prediction and phishing_prediction == -1:
            #     return render_template('notspambutphishing.html')
            # elif 1 in prediction and phishing_prediction == 1:
            #     return render_template('spambutnotphishing.html')
            # elif 0 in prediction and phishing_prediction == 1:
            #     return render_template('hamandnotphishing.html')
            # elif 1 in  prediction :
            #     return render_template('spam.html')
            # else:
            #     return render_template('ham.html')
            

    except Exception as e:
        return render_template('index.html', prediction=None, error=str(e))

    return render_template('spam.html', prediction=prediction,phishing_prediction=phishing_prediction)
               

if __name__ == '__main__':
    app.run(debug=True)
