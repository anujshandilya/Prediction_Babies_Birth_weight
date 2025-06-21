from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

def clean_data(birth_data):

    gestation = float(birth_data['gestation'])
    parity = bool(birth_data['parity'])
    age = int(birth_data['age'])
    height = float(birth_data['height'])
    weight = float(birth_data['weight'])
    smoke = bool(birth_data['smoke'])

    cleaned_data = {"gestation":[gestation],
                  "parity":[parity],
                  "age":[age],
                  "height":[height],
                  "weight":[weight],
                  "smoke":[smoke]}

    return cleaned_data


@app.route('/', methods = ['GET'])
def home():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.form
    data = clean_data(data)
    dt = pd.DataFrame(data)
    model = pickle.load(open("Model/model.pkl","rb"))
    pred = model.predict(dt)
    pred = round(float(pred),2)
    
    return render_template("predict.html", pred = pred)



if __name__ == '__main__':
    app.run(debug=True)