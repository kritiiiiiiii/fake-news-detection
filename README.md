## Fake News Detection System

A Machine Learning-powered Fake News Detection application that classifies news articles as Real or Fake using Natural Language Processing (NLP), TF-IDF Vectorization, and Logistic Regression.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)

## Live Demo

🚀 Try the application here: 
https://fake-news-detection-hfugwfwsh9cjss79k3dmdl.streamlit.app

## Application  Screenshots

### Home Page

![Home Page](screenshots/home.png)

### Prediction Result

![Prediction Result](screenshots/prediction-result.png)

### Detailed Analysis

![Analysis](screenshots/analysis.png)

### Model Evaluation

![Confusion Matrix](screenshots/confusion-matrix.png)

## Features

- Real-time Fake News Classification
- NLP-based Text Preprocessing
- TF-IDF Feature Engineering
- Logistic Regression Classification
- Confidence Score Visualization
- Keyword Importance Analysis
- Dataset Statistics Dashboard
- Confusion Matrix Evaluation
- Interactive Streamlit Interface

## Workflow

1. User enters a news article.
2. Text is cleaned and preprocessed.
3. TF-IDF vectorization converts text into numerical features.
4. Logistic Regression predicts whether the news is Real or Fake.
5. Confidence scores and influential keywords are displayed.

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Streamlit
* Matplotlib
* Seaborn
* Plotly

## Project Structure

```text
fake-news-detection/
│
├── app/
│   └── app.py
│
├── src/
│   ├── preprocess.py
│   ├── vectorizer.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── models/
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── metrics.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── screenshots/
│   ├── home.png
│   ├── prediction-result.png
│   ├── analysis.png
│   └── confusion-matrix.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### Clone Repository

```bash
git clone https://github.com/kritiiiiiiii/fake-news-detection.git
```

### Navigate to Project Directory

```bash
cd fake-news-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit Application

```bash
streamlit run app/app.py
```

## Model Performance

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 98.24% |
| Naive Bayes | 94.67% |

The Logistic Regression model was selected as the final model due to its superior performance.

## Dataset

The project uses a public Fake News Detection dataset containing:

- 44,898 news articles
- 21,417 Real news articles
- 23,481 Fake news articles

> Note: The original dataset is not included in this repository.

## Future Improvements

- News URL analysis

- News source credibility scoring

- Deep Learning models (LSTM/BERT)

- Mobile-responsive UI enhancements

- Multi-language support

## Author

**Kriti Jha**

- GitHub: [@kritiiiiiiii](https://github.com/kritiiiiiiii)
- LinkedIn: [Kriti Jha](https://www.linkedin.com/in/kriti-jha-9b8435323/)


---
