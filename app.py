# app.py
import streamlit as st
from datetime import datetime
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Use TextBlob (simple, light) for sentiment
from textblob import TextBlob

DB_PATH = "journal.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        text TEXT,
        sentiment_label TEXT,
        sentiment_score REAL,
        emotion_label TEXT,
        keywords TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_entry(text, sentiment_label, sentiment_score, emotion_label, keywords):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO entries (timestamp, text, sentiment_label, sentiment_score, emotion_label, keywords)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), text, sentiment_label, sentiment_score, emotion_label, keywords))
    conn.commit()
    conn.close()

def load_entries():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM entries ORDER BY id DESC", conn)
    conn.close()
    return df

def extract_keywords(text, top_k=5):
    words = [w.strip(".,!?()[]{}\"'").lower() for w in text.split()]
    stop = set(["i","to","the","a","and","is","it","of","in","on","my","me","for","that","with","was","are","had","be","as"])
    freq = {}
    for w in words:
        if not w or w in stop or len(w) < 3: continue
        freq[w] = freq.get(w,0)+1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ", ".join([i[0] for i in items])

def analyze_text(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1 (neg) to +1 (pos)
    if polarity > 0.2:
        sentiment_label = "POSITIVE"
    elif polarity < -0.2:
        sentiment_label = "NEGATIVE"
    else:
        sentiment_label = "NEUTRAL"
    sentiment_score = abs(polarity)
    if polarity > 0.3:
        emotion_label = "joy"
    elif polarity < -0.3:
        emotion_label = "sadness"
    else:
        emotion_label = "neutral"
    keywords = extract_keywords(text)
    return sentiment_label, sentiment_score, emotion_label, keywords

def recommend(sentiment_label, emotion_label):
    recs = []
    if sentiment_label == "NEGATIVE" or emotion_label in ["sadness"]:
        recs.append("Try a 3-minute deep breathing exercise.")
        recs.append("Write 3 things you are grateful for.")
    elif sentiment_label == "POSITIVE":
        recs.append("Nice — note what made today good and try to repeat it.")
    else:
        recs.append("Take a short walk or a 5-minute break.")
    return recs

# Initialize DB
init_db()

# Streamlit UI
st.set_page_config(page_title="Insight — AI Wellness Journal", layout="centered")
st.title("Insight — AI Wellness Journal (Beginner MVP)")

with st.form("journal_form"):
    st.subheader("Write your journal entry")
    text = st.text_area("How are you feeling today? (write freely)", height=180)
    submitted = st.form_submit_button("Analyze & Save")

if submitted:
    if not text.strip():
        st.warning("Please type something before saving.")
    else:
        sentiment_label, sentiment_score, emotion_label, keywords = analyze_text(text)
        save_entry(text, sentiment_label, sentiment_score, emotion_label, keywords)
        st.success("Entry saved.")
        st.markdown(f"**Sentiment:** {sentiment_label} ({sentiment_score:.2f})")
        st.markdown(f"**Emotion:** {emotion_label}")
        st.markdown(f"**Keywords:** {keywords}")
        with st.expander("Personalized suggestions"):
            for r in recommend(sentiment_label, emotion_label):
                st.write("- " + r)

st.subheader("Recent entries")
df = load_entries()
if df.empty:
    st.info("No entries yet. Add your first entry above.")
else:
    st.dataframe(df[['timestamp','text','sentiment_label','emotion_label','sentiment_score']].head(10))

    # Plot sentiment score over time
    df_plot = df.copy()
    df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
    df_plot = df_plot.sort_values('timestamp')
    fig, ax = plt.subplots()
    ax.plot(df_plot['timestamp'], df_plot['sentiment_score'], marker='o')
    ax.set_xlabel("Time")
    ax.set_ylabel("Sentiment score (0-1)")
    ax.set_title("Sentiment over time")
    plt.xticks(rotation=30)
    st.pyplot(fig)

st.markdown("---")
st.write("Data is stored locally in `journal.db` in this folder.")
