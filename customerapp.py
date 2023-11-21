import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas
import time
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = joblib.load(open('D:\\College\\AI_Extership\\DecisionTree.joblib', 'rb'))
scale = joblib.load(open('D:\\College\\AI_Extership\\MinMaxScaler.joblib', 'rb'))

@app.route('/')
def home():
    return render_template('CustomerSegmentation.html')

@app.route('/predict',methods = ["POST","GET"])
def predict():
    input_feature = [float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]
    names = ['Sex', 'Marital status', 'Age', 'Education', 'Income','Occupation', 'Settlement size']
    data = pandas.DataFrame(features_values,columns=names)
    data_scaled = scale.transform(data)
    prediction = model.predict(data_scaled)
    return render_template('prediction.html', data=prediction)
    
if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')