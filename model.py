import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Sample training data
data = {
    "tenure": [1, 5, 10, 20, 30, 40],
    "monthly": [50, 60, 70, 80, 90, 100],
    "total": [50, 300, 700, 1600, 2700, 4000],
    "churn": [1, 1, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

X = df[["tenure", "monthly", "total"]]
y = df["churn"]

model = LogisticRegression()
model.fit(X, y)

# Save model
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully")