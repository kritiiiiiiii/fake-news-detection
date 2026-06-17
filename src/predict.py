import joblib
from src.preprocess import clean_text

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def get_top_words(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    
    feature_names = vectorizer.get_feature_names_out()
    
    indices = vectorized.toarray()[0].argsort()[-5:]
    words = [feature_names[i] for i in indices]
    
    return words

def predict_news(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    pred = model.predict(vectorized)[0]

    proba = model.predict_proba(vectorized)[0]

    fake_prob = proba[0]
    real_prob = proba[1]

    confidence = max(proba)

    label = "Real News" if pred == 1 else "Fake News"

    top_words = get_top_words(text)

    return (
        label,
        confidence,
        top_words,
        real_prob,
        fake_prob
    )

def get_global_top_words():

    feature_names = vectorizer.get_feature_names_out()

    coefs = model.coef_[0]

    top_real_idx = coefs.argsort()[-5:]
    top_fake_idx = coefs.argsort()[:5]

    real_words = [feature_names[i] for i in top_real_idx]
    real_scores = [coefs[i] for i in top_real_idx]

    fake_words = [feature_names[i] for i in top_fake_idx]
    fake_scores = [abs(coefs[i]) for i in top_fake_idx]

    return (
        real_words,
        real_scores,
        fake_words,
        fake_scores
    )

if __name__ == "__main__":
    Sample = "Breaking: Government announces new policy today."
    print(predict_news(Sample))
