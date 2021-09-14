

from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import joblib
import pandas as pd


app = Flask(__name__)

model = joblib.load("covid_symp.pk1")




@app.route('/')
def hello_world():
    return render_template("test_index.html")


@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        Fever = request.form["Fever"]
        #Tiredness = request.form["Tiredness"]
        Dry_Cough = request.form["Dry-Cough"]
        Difficulty_in_Breathing = request.form["Difficulty_in_Breathing"]
        Sore_Throat = request.form["Sore-Throat"]
        head_ache = request.form["head_ache"]
        gender = request.form["gender"]
        #Pains = request.form["Pains"]
        #Nasal_Congestion = request.form["Nasal-Congestion"]
        #Runny_Nose = request.form["Runny-Nose"]
        #Diarrhea = request.form["Diarrhea"]
        Age = request.form["AGE"]
        #Person_in_Contact = request.form["Person_in_Contact"]


       
        prediction=model.predict([[
            Dry_Cough,
            Fever,
            Sore_Throat,
            Difficulty_in_Breathing,
            head_ache,
            Age,
            gender
            ]])

        output=prediction[0]

        if output==1:
            return render_template('test_covid.html',prediction_text="You have high chances of covid")

        else:
            return render_template('test_covid.html',prediction_text="You have low chances of covid")
    
    return render_template('wrong_credentials.html')
       
        



if __name__ == "__main__":
    app.run(debug=True)
    