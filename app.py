import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
import psycopg2
from flask_cors import CORS, cross_origin

app = Flask(__name__)
model = pickle.load(open('./ModelSaving/model.pkl', 'rb'))
print("Inside model")
scalar = pickle.load(open('./DataScaler/Scaler.pkl', 'rb'))
print("inside scalar")

@cross_origin()
@app.route('/', methods=['GET'])
def home():
    print("Inside home page")
    return render_template('./home.html')

@cross_origin()
@app.route('/info', methods=['GET'])
def info():
    print("Inside info page")
    return render_template('./info.html')

@cross_origin()
@app.route('/developer', methods=['GET'])
def developer():
    print("Inside home page")
    return render_template('./developer.html')

@cross_origin()
@app.route('/contact', methods=['GET'])
def contact():
    print("Inside contact page")
    return render_template('./contact.html')

@cross_origin()
@app.route('/app', methods=['GET'])
def index_page():
    print("Inside app")
    return render_template('./index.html')

@cross_origin()
@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        cement = float(request.form['cement'])

        slag = float(request.form['Blast Furnace Slag'])

        ash = float(request.form['Fly Ash'])

        water = float(request.form['Water'])

        sup = float(request.form['SuperPlasticizer'])

        co = float(request.form['Coarse Aggregate'])

        fine = float(request.form['Fine Aggregate'])

        week = str(request.form['week'])
        if week == '13weeks':
            week = 1, 0, 0, 0, 0, 0
        elif week == '2weeks':
            week = 0, 1, 0, 0, 0, 0
        elif week == '3':
            week = 0, 0, 1, 0, 0, 0
        elif week == '4weeks':
            week = 0, 0, 0, 1, 0, 0
        elif week == '8weeks':
            week = 0, 0, 0, 0, 1, 0
        elif week == 'moreweeks':
            week = 0, 0, 0, 0, 0, 1
        else:
            week = 0, 0, 0, 0, 0, 0

        col = ([[cement, slag, ash, water, sup, co, fine, *week]])
        print(col)
        scl = scalar.transform(col)
        print(scl)
        pred = model.predict(scl)
        print(pred)
        return render_template('./result.html', Prediction_text = f'The Strength of Concrete is {round(pred[0],2)}')
    else:
        return  render_template('./home.html')


if __name__ == "__main__":
    app.run(debug=True)
