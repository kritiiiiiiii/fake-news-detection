import streamlit as st
import sys
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load metrics
metrics = joblib.load("models/metrics.pkl")

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_news, get_global_top_words

def highlight_text(text, words, color):
    for w in words:
        text = text.replace(
            w,
            f"<span style='background-color:{color}; padding:2px 4px; border-radius:4px'>{w}</span>"
        )
    return text

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        text-align: center;
        font-weight: 700;
    }
    .stTextArea textarea {
        border-radius: 10px;
        padding: 10px;
        font-size: 15px;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("Fake News Detection System")
st.caption("Analyze news content using NLP and Machine Learning")

st.markdown("###")  # spacing

# ------------------ METRICS ------------------
st.markdown("#### Model Performance")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.metric(
        "Logistic Regression Accuracy",
        f"{metrics['lr_accuracy']*100:.2f}%"
    )

with col2:
    st.metric(
        "Naive Bayes Accuracy",
        f"{metrics['nb_accuracy']*100:.2f}%"
    )

st.divider()

# ------------------ DATASET STATS ------------------

st.markdown("#### Dataset Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Articles", "44,898")

with col2:
    st.metric("Real", "21,417")

with col3:
    st.metric("Fake", "23,481")

st.divider()

# ------------------ MODEL INSIGHTS ------------------
st.markdown("#### Learned Dataset Features")

st.caption("These reflect dataset patterns, not factual truth signals.")

real_words, real_scores, fake_words, fake_scores = get_global_top_words()

df_real = pd.DataFrame({
    "Word": real_words,
    "Score": real_scores
})

fig = px.bar(
    df_real,
    x="Score",
    y="Word",
    orientation="h",
    title="Top Features Associated with Real News"
)

fig.update_layout(
    template="plotly_dark",
    showlegend=False,
    yaxis={"categoryorder": "total ascending"},
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig, use_container_width=True)

df_fake = pd.DataFrame({
    "Word": fake_words,
    "Score": fake_scores
})

fig = px.bar(
    df_fake,
    x="Score",
    y="Word",
    orientation="h",
    title="Top Features Associated with Fake News"
)

fig.update_layout(
    template="plotly_dark",
    showlegend=False,
    yaxis={"categoryorder": "total ascending"},
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig, use_container_width=True)

st.info("Note: Predictions may rely on stylistic and source-specific patterns due to dataset bias, rather than purely semantic truth evaluation.")

st.divider()

# ------------------ INPUT ------------------
st.markdown("#### Input")

mode = st.selectbox(
    "Input Type",
    ["News Article", "Short Claim"]
)

if mode == "News Article":
    placeholder = "Paste a full news article..."

else:
    placeholder = "Example: Coffee cures all diseases"

user_input = st.text_area(
    "Enter content",
    placeholder=placeholder
)
st.markdown("###")  # spacing

if user_input.strip():
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Word Count", len(user_input.split())
       )  
        
    with col2:
        st.metric("Character Count", len(user_input)
         )
        
    if len(user_input.split()) < 10:
        st.warning(
            "Very short inputs may lead to unreliable predictions. For best results, please provide a more substantial text (at least 10 words)."
        )

st.markdown("###")  # spacing
        

# ------------------ HISTORY ------------------

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ BUTTON ------------------
# ------------------ BUTTON ------------------
if st.button("Analyze News"):
    if user_input.strip() != "":
        with st.spinner("Analyzing..."):
            result, confidence, top_words, real_prob, fake_prob = predict_news(user_input)

    # Save prediction

        st.session_state.history.insert(
            0,
            {
                "result": result,
                "confidence": confidence
            }
        )
    # Keep only latest 5

        st.session_state.history = st.session_state.history[:5]

        # RESULT DISPLAY
        st.markdown(f"### Classification Result: {result}")

        if confidence >= 0.85:
            st.success(f"High Confidence ({confidence*100:.1f}%)")

        elif confidence >= 0.65:
            st.warning(f"Moderate Confidence ({confidence*100:.1f}%)")

        else:
            st.error(f"Low Confidence ({confidence*100:.1f}%)")

        st.caption(f"Confidence Score: {confidence*100:.1f}%")
        st.progress(float(confidence))

        st.markdown("#### Prediction Probabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Real News",
                f"{real_prob*100:.1f}%"
         )

        with col2:
            st.metric(
                 "Fake News",
                 f"{fake_prob*100:.1f}%"
            )

        # KEY WORD CHIPS
        st.markdown("#### Most Influential Features")
        st.caption("Words/Phrases that contributed most to the prediction")

        chip_html = ""

        for word in top_words:
            chip_html += f"""
            <span style="
                background-color:#1f2937;
                color:#e5e7eb;
                padding:6px 10px;
                border-radius:12px;
                margin:4px;
                display:inline-block;
                font-size:13px;
            ">
                {word}
            </span>
            """

        st.markdown(chip_html, unsafe_allow_html=True)

        # HIGHLIGHTED ANALYSIS

        color = "#22c55e" if "Real" in result else "#ef4444"

        highlighted = highlight_text(
            user_input.lower(),
            [w.lower() for w in top_words],
            color
        )

        st.markdown("#### Highlighted Analysis")
        st.markdown(highlighted, unsafe_allow_html=True)

        # FINAL RESULT MESSAGE

        if "Real" in result:
            st.success("Predicted Class: Real News")
        else:
            st.error("Predicted Class: Fake News")

        report = f"""
        Prediction Report

        Prediction: {result}
        
        Confidence: {confidence*100:.1f}%
        
        Real Probability: {real_prob*100:.1f}%
        
        Fake Probability: {fake_prob*100:.1f}%
        
        Top Influential Features:
        {", ".join(top_words)}
        
        Input Text:
        {user_input}"""
        
        st.download_button(
            label="Download Report",data=report,file_name="prediction_report.txt",mime="text/plain"
        )

    else:
        st.warning("Please enter some text")

st.divider()

st.markdown("#### Recent Predictions")

for item in st.session_state.history:

    emoji = "✅" if "Real" in item["result"] else "❌"

    st.write(
        f"{emoji} {item['result']} ({item['confidence']*100:.1f}%)"
    )
    
st.divider()

st.markdown("#### Model Evaluation (Test Dataset)")

conf_matrix = metrics["conf_matrix"]

fig, ax = plt.subplots(figsize=(5,4))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"]
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

st.divider()