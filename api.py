from fastapi import FastAPI
import pickle

app = FastAPI()

# Load model + vectorizer
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(text: str):
    vec = vectorizer.transform([text])
    emotion = model.predict(vec)[0]
    return {"emotion": emotion}

