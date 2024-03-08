import streamlit as st
from Prototype_01 import AudioAnalyzer  # Import your script that contains the AudioAnalyzer class
from emotion_prediction import emotion_prediction
import nltk
import pandas as pd
import base64 
import numpy as np 
import librosa
nltk.download('punkt')  # Download the punkt tokenizer data
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder

# Function to perform text analysis
def text_analysis_page():
    st.title("Text Analysis")
    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_audio is not None:

        # Display the audio player for the uploaded file
        st.audio(uploaded_audio, format='audio/wav', start_time=0)
        
        # Create an instance of the AudioAnalyzer class
        analyzer = AudioAnalyzer()

        # Analyze the uploaded audio file
        st.text("Analyzing the uploaded audio file...")
        with st.spinner("Analyzing..."):
            speech_to_text = analyzer.audio_to_text(uploaded_audio)
            sentiment_on_text = analyzer.sent_on_text(speech_to_text)
            summarized_text = analyzer.text_summarizer(speech_to_text)
            insights_from_text = analyzer.insights_from_gpt(speech_to_text)

        # Convert the sentiment result to a DataFrame
        sentiment_df = pd.DataFrame(sentiment_on_text)

        # Create a DataFrame for export
        export_data = {
            "Text from Audio File": [speech_to_text],
            "Sentiment Label": sentiment_df["label"].values,
            "Sentiment Score": sentiment_df["score"].values,
            "Summarized Text": [summarized_text],
            "Valuable Insights": [insights_from_text]
        }
        export_df = pd.DataFrame(export_data)

        # Display the results on the Streamlit web page
        st.subheader("Text from Audio File:")
        sentences = nltk.sent_tokenize(speech_to_text)
        for sentence in sentences:
            st.write(sentence)


        st.subheader("Sentiment of the Text:")
        sentiment_label = sentiment_on_text[0]['label']
        sentiment_score = sentiment_on_text[0]['score']
        st.text(f"Label: {sentiment_label}")
        st.text(f"Confidence Score: {sentiment_score}")

        st.subheader("Summarized Text:")
        sentences = nltk.sent_tokenize(summarized_text)
        for sentence in sentences:
            st.text(sentence)

        st.subheader("Valuable Insights:")
        st.text(insights_from_text)
        # Add an export button to download the data as a CSV file
        if st.button("Export Data as CSV"):
            st.write("Exporting data...")
            st.markdown(get_table_download_link(export_df), unsafe_allow_html=True)

    # Your existing code for text analysis goes here

# Function to perform audio analysis
def audio_analysis_page():
    st.title("Audio Analysis")
    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_audio is not None:

        # Analyze the uploaded audio file
        st.text("Analyzing the uploaded audio file...")
        with st.spinner("Analyzing..."):
            # Display the audio player for the uploaded file
            st.audio(uploaded_audio, format='audio/wav', start_time=0)

            emotion_predictor = emotion_prediction()

            cleaned_label = emotion_predictor.make_prediction(uploaded_audio)
            st.subheader(f"Predicted Emotion: {cleaned_label}")

# Function to download CSV link
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="audio_analysis.csv">Download CSV</a>'
    return href

def main():
    st.title("Sentiment Analyzer")

    st.markdown("""
    ### Welcome to the Sentiment Analyzer app. Choose the type of analysis you want to perform:
    """)

    # Add text descriptions above buttons for Text Analysis and Audio Analysis
    st.write("Audio to Text Analysis:")
    text_analysis_button = st.button("Text Analysis")

    st.write("\nAudio Analysis:")
    audio_analysis_button = st.button("Audio Analysis")

    # Handle button clicks
    if text_analysis_button:
        # Redirect to the Text Analysis page
        st.experimental_set_query_params(page="text_analysis")
        
    if audio_analysis_button:
        # Redirect to the Audio Analysis page
        st.experimental_set_query_params(page="audio_analysis")

    # Check if the page query parameter is set and navigate to the corresponding page
    current_page = st.experimental_get_query_params().get("page", [""])[0]
    if current_page == "text_analysis":
        text_analysis_page()
    elif current_page == "audio_analysis":
        audio_analysis_page()

if __name__ == "__main__":
    main()
