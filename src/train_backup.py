import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocess import clean_text

# ensure models folder exists
os.makedirs("models", exist_ok=True)

# LOAD DATA
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real])
df = df.sample(frac=1).reset_index(drop=True)

# CLEAN TEXT
df["text"] = df["text"].apply(clean_text)

# TF-IDF with n-grams
tfidf = TfidfVectorizer(max_df=0.7, ngram_range=(1,2))
X = tfidf.fit_transform(df["text"])
y = df["label"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODELS
lr = LogisticRegression(max_iter=1000)
nb = MultinomialNB()

# TRAIN
lr.fit(X_train, y_train)
nb.fit(X_train, y_train)

# PREDICT
lr_pred = lr.predict(X_test)
nb_pred = nb.predict(X_test)

cm = confusion_matrix(y_test, lr_pred)

# EVALUATE
lr_acc = accuracy_score(y_test, lr_pred)
nb_acc = accuracy_score(y_test, nb_pred)

print("🔹 Logistic Regression Accuracy:", lr_acc)
print("🔹 Naive Bayes Accuracy:", nb_acc)

print("\nLogistic Regression Report:\n", classification_report(y_test, lr_pred))

# SAVE METRICS
metrics = {
    "lr_accuracy": lr_acc,
    "nb_accuracy": nb_acc,
    "conf_matrix": cm
}
joblib.dump(metrics, "models/metrics.pkl")

# SAVE MODEL + VECTORIZER
print("\nSaving Logistic Regression model ...")
joblib.dump(lr, "models/model.pkl")
joblib.dump(tfidf, "models/vectorizer.pkl")

print("Model + metrics saved successfully!")