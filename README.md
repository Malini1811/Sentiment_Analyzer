#  Multilanguage Sentiment Analyzer Web App

This is a **Streamlit-based web application** that performs **paragraph-level and word-level sentiment analysis** on multilingual input text. It uses a pre-trained HuggingFace Transformer model (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) to classify input as **Positive**, **Neutral**, or **Negative**.

##  Project Structure

```
multilingual-sentiment-analyzer/
│
├── app.py                    # Streamlit UI app
├── analyzer.py               # Sentiment analysis functions using Transformers
├── requirements.txt          # Required Python packages
└── README.md                 # Project documentation
```

##  How to Run the App

```bash
streamlit run app.py
```

Then, visit `http://localhost:8501/` in your browser.

## Input Text for Sentiment Prediction

You can enter any sentence, phrase, or paragraph in any supported language. The app will return:

* Overall sentiment probabilities (Positive, Neutral, Negative)
* Bar chart for sentiment distribution
* Word-by-word classification

##  Model Details

* **Model Name:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`
* **Type:** Multilingual Transformer
* **Framework:** HuggingFace Transformers + PyTorch
* **Supports:** Multiple languages
* **Granularity:** Paragraph-level and word-level sentiment

##  Notes

* Ideal for analyzing reviews, comments, tweets, and short texts
* Multilingual capability makes it suitable for global use cases
* Easy to extend or modify the Streamlit UI

