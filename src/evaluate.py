import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from preprocess import clean_text

# LOAD MODEL
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# LOAD DATA
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real])
df = df.sample(frac=1).reset_index(drop=True)

df["text"] = df["text"].apply(clean_text)

X = vectorizer.transform(df["text"])
y = df["label"]

# PREDICT
y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred)

print("Confusion Matrix:\n", cm)