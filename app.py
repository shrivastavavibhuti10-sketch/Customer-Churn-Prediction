from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("churn_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    tenure = float(request.form["tenure"])
    monthly = float(request.form["monthly"])
    total = float(request.form["total"])

    prediction = model.predict([[tenure, monthly, total]])[0]

    if prediction == 1:
        result = "Customer Will Churn ❌"
    else:
        result = "Customer Will Stay ✅"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)