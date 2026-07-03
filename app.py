from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("decision_tree.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/submit", methods=["POST"])
def submit():

    prediction = model.predict([[0, 1, 0, 1, 0, 5000, 2000, 150, 360, 1, 2]])

    if prediction[0] == 1:
        result = "Loan Approved ✅"
    else:
        result = "Loan Rejected ❌"

    return render_template("output.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)