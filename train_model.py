import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path, sep=";", names=["text", "emotion"])
    return df

train_df = load_data("data/train.txt")

# Convert text to vectors
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df["text"])
y_train = train_df["emotion"]

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model + vectorizer
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model training completed!")
