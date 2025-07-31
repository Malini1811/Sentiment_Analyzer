import streamlit as st
from analyzer import analyze_sentiment_paragraph, analyze_words_sentiment_multilingual
import matplotlib.pyplot as plt

st.set_page_config(page_title="Multilanguage Sentiment Analyzer", layout="centered")
st.title(" Multilanguage Sentiment Analyzer")

user_input = st.text_area("Enter the sentence :")

if st.button("Analyze Sentiment"):
    if user_input:
        # Paragraph-Level Sentiment
        result = analyze_sentiment_paragraph(user_input)
        st.subheader("Overall Paragraph Sentiment:")
        for sentiment, score in result.items():
            emoji = "😊" if sentiment == "Positive" else "😐" if sentiment == "Neutral" else "😠"
            st.write(f"{emoji} **{sentiment}: {score}**")

        # Bar Chart
        st.subheader("Sentiment Distribution Chart:")
        fig, ax = plt.subplots()
        ax.bar(result.keys(), result.values(), color=["red", "gray", "green"])
        ax.set_ylabel("Score")
        ax.set_title("Paragraph-Level Sentiment")
        st.pyplot(fig)

        # Word-Level Sentiment
        st.subheader("Word-Level Sentiment Analysis")
        word_result = analyze_words_sentiment_multilingual(user_input)
        st.success(f"😊 Positive Words: {', '.join(word_result['Positive']) or 'None'}")
        st.error(f"😠 Negative Words: {', '.join(word_result['Negative']) or 'None'}")
        st.info(f"😐 Neutral Words: {', '.join(word_result['Neutral']) or 'None'}")

    else:
        st.warning("⚠️ Please enter some text.")
