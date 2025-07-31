from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Label mapping for the model
labels = ["Negative", "Neutral", "Positive"]

# Paragraph-Level Sentiment Analysis
def analyze_sentiment_paragraph(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.softmax(dim=1)[0]

    result = {
        "Negative": round(scores[0].item(), 2),
        "Neutral": round(scores[1].item(), 2),
        "Positive": round(scores[2].item(), 2)
    }
    return result

def analyze_words_sentiment_multilingual(text):
    words = text.strip().split()
    sentiment_words = {"Positive": [], "Negative": [], "Neutral": []}

    for word in words:
        inputs = tokenizer(word, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits.softmax(dim=1)[0]
            predicted_label = labels[scores.argmax().item()]
            sentiment_words[predicted_label].append(word)

    return sentiment_words
