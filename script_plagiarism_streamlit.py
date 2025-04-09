import streamlit as st
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

# ---------- Preprocessing ----------
def preprocess_script(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower() for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(tokens)

# ---------- Vectorization ----------
def vectorize_scripts(scripts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(scripts)

# ---------- Similarity Calculation ----------
def calculate_similarity(tfidf_matrix):
    return cosine_similarity(tfidf_matrix)

# ---------- Visualization ----------
def display_heatmap(similarity_matrix, filenames):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_matrix, xticklabels=filenames, yticklabels=filenames,
                cmap="YlGnBu", annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)

# ---------- Main App ----------
st.title("🎬 Script Plagiarism Detector")
st.write("Upload multiple movie scripts in `.txt` format to compare their similarity.")

uploaded_files = st.file_uploader("Choose script files", type="txt", accept_multiple_files=True)

if uploaded_files:
    raw_scripts = []
    filenames = []

    for file in uploaded_files:
        text = file.read().decode("utf-8")
        raw_scripts.append(preprocess_script(text))
        filenames.append(file.name)

    st.success("Scripts uploaded and preprocessed successfully.")

    with st.spinner("Calculating similarity..."):
        tfidf_matrix = vectorize_scripts(raw_scripts)
        similarity_matrix = calculate_similarity(tfidf_matrix)

    st.subheader("🔍 Similarity Heatmap")
    display_heatmap(similarity_matrix, filenames)

    st.subheader("⚠️ Potential Plagiarism Cases")
    threshold = 0.75
    flagged = []
    n = len(filenames)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] > threshold:
                flagged.append((filenames[i], filenames[j], similarity_matrix[i][j]))

    if flagged:
        for file1, file2, score in flagged:
            st.write(f"**{file1}** ↔ **{file2}** : Similarity Score = {score:.2f}")
    else:
        st.info("No suspicious similarity scores found above threshold.")
