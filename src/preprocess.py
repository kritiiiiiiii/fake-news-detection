import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()

    # 🔥 REMOVE DATASET BIAS WORDS (ADD HERE)
    text = re.sub(r'reuters', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'via', '', text)
    text = re.sub(r'featured', '', text)
    text = re.sub(r'washington', '', text)

    # existing cleaning
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)


#test block
if __name__ == "__main__":
    import pandas as pd

    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")

    fake["label"] = 0
    real["label"] = 1

    df = pd.concat([fake, real])
    df = df.sample(frac=1).reset_index(drop=True)

    df["text"] = df["text"].apply(clean_text)

    print("Preprocessing Done!")
    print(df[["text", "label"]].head())