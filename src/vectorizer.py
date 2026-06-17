import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import clean_text

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real])
df = df.sample(frac=1).reset_index(drop=True)

df["text"] = df["text"].apply(clean_text)

tfidf = TfidfVectorizer(max_df = 0.7, ngram_range=(1,2), max_features=10000)

X = tfidf.fit_transform(df["text"])
y = df["label"]

print("Vectorization Done")
print("Shape of X: ", X.shape)