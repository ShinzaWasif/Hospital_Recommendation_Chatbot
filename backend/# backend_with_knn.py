# backend_with_knn.py
from flask import Flask, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from fuzzywuzzy import process
import json
from flask_cors import CORS
import re
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import numpy as np
from word2number import w2n
import time
import random
import os
import pygame
from gtts import gTTS
import threading
import inflect 
import pyttsx3
from collections import Counter
import pandas as pd


# NEW ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score   # <-- add this line



app = Flask(__name__)
CORS(app)

nltk.download("punkt")
nltk.download("punkt_tab")

# Load hospital data
with open("hospitals.json", "r", encoding="utf-8") as file:
    hospital_data = json.load(file)

# Load Wav2Vec 2.0 model (if you need audio transcription)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

def convert_numbers(transcription):
    words = transcription.lower()
    try:
        return str(w2n.word_to_num(words))
    except ValueError:
        return words

def extract_number_range(text):
    match = re.findall(r"(\d+[\s-]\d[\s-]\d)", text)
    return match if match else None

def transcribe_audio(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    max_length = 50
    chunk_size = sr * max_length
    transcriptions = []

    for i in range(0, len(speech), chunk_size):
        chunk = speech[i: i + chunk_size]
        input_values = processor(chunk, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions.append(processor.batch_decode(predicted_ids)[0])

    full_transcription = " ".join(transcriptions).lower()
    full_transcription = convert_numbers(full_transcription)
    number_ranges = extract_number_range(full_transcription)
    if number_ranges:
        full_transcription += " " + " ".join(number_ranges)

    return full_transcription

def extract_city_from_query(query):
    query_tokens = set(word_tokenize(query.lower()))
    possible_cities = {hospital["City"].lower() for hospital in hospital_data if "City" in hospital}
    for city in possible_cities:
        if city in query_tokens:
            return city
    return None

def extract_fee_from_query(query):
    # match "20000 - 40000" (allow spaces around dash)
    range_match = re.search(r"(\d{2,6})\s*-\s*(\d{2,6})", query)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))

    # single number like 30000 or 50k
    single_match = re.search(r"(\d{2,6})", query)
    if single_match:
        fee = int(single_match.group(1))
        return fee, fee

    # 🚨 always return tuple, never None
    return None, None

def is_within_fee_range(hospital_fee, min_fee, max_fee):
    m = re.search(r"(\d{2,6})\s*-\s*(\d{2,6})", hospital_fee)
    if m:
        hmin, hmax = int(m.group(1)), int(m.group(2))
        # overlap check
        return not (hmax < min_fee or hmin > max_fee)
    # single value case
    m2 = re.search(r"(\d{2,6})", hospital_fee)
    if m2:
        val = int(m2.group(1))
        return min_fee <= val <= max_fee
    return False


def find_matching_hospitals(query):
    query = query.lower().strip()
    query_tokens = set(word_tokenize(query))
    city_filter = extract_city_from_query(query)
    min_fee, max_fee = extract_fee_from_query(query)

    exact_matches = []
    fuzzy_candidates = []
    specializations_dict = {}

    for hospital in hospital_data:
        specialization = hospital.get("Specialization", "").lower()
        hospital_city = hospital.get("City", "").lower()
        hospital_fee = hospital.get("Fees", "")

        specializations_dict[specialization] = hospital
        specialization_tokens = set(word_tokenize(specialization))

        if query in specialization or query_tokens & specialization_tokens:
            if city_filter and hospital_city != city_filter:
                continue
            if min_fee is not None and not is_within_fee_range(hospital_fee, min_fee, max_fee):
                continue
            exact_matches.append(hospital)

    if not exact_matches:
        fuzzy_results = process.extract(query, specializations_dict.keys(), limit=10)
        for best_match, score in fuzzy_results:
            if score > 70:
                hospital = specializations_dict[best_match]
                hospital_city = hospital.get("City", "").lower()
                hospital_fee = hospital.get("Fees", "")
                if city_filter and hospital_city != city_filter:
                    continue
                if min_fee is not None and not is_within_fee_range(hospital_fee, min_fee, max_fee):
                    continue
                fuzzy_candidates.append(hospital)

    final_results = exact_matches + fuzzy_candidates
    try:
        return sorted(final_results, key=lambda h: int(re.search(r"(\d+)", h.get("Fees","0")).group(1))) if final_results else None
    except Exception:
        return final_results if final_results else None

p = inflect.engine()

def convert_numbers_to_words(text):
    def replace_number(match):
        num = int(match.group(0))
        return p.number_to_words(num, andword="")
    return re.sub(r"\b\d+\b", replace_number, text)

audio_lock = threading.Lock()

def speak(hospitals, user_fee_range=None, use_offline_tts=True):
    def tts_worker():
        try:
            if not hospitals:
                text = "Sorry, no matching hospitals found."
            else:
                hospital_info = []
                for hospital in hospitals:
                    name = hospital.get("Name", "Unknown Hospital")
                    city = hospital.get("City", "Unknown Location")
                    hospital_fees = hospital.get("Fees", "Not Available")
                    hospital_fees = re.sub(r"\b0+\b", "zero", hospital_fees)
                    hospital_fees = convert_numbers_to_words(hospital_fees)
                    user_fee_text = ""
                    if user_fee_range:
                        min_fee, max_fee = user_fee_range
                        user_fee_text = f"Your fee range is {convert_numbers_to_words(str(min_fee))} to {convert_numbers_to_words(str(max_fee))}."
                    hospital_text = f"{name}, located in {city}. Hospital fee range is {hospital_fees}. {user_fee_text}"
                    hospital_info.append(hospital_text)
                text = "Dear user, according to your search following are the hospitals. " + " ".join(hospital_info) + " For more, please visit the website."

            if use_offline_tts:
                engine = pyttsx3.init()
                voices = engine.getProperty("voices")
                indian_voice = None
                for voice in voices:
                    if "en-in" in voice.id.lower() or "india" in voice.name.lower():
                        indian_voice = voice.id
                        break
                if indian_voice:
                    engine.setProperty("voice", indian_voice)
                engine.setProperty("rate", 165)
                with audio_lock:
                    engine.say(text)
                    engine.runAndWait()
            else:
                tts = gTTS(text, lang="en", tld="co.in")
                temp_filename = f"temp_audio_{int(time.time())}_{random.randint(1000, 9999)}.mp3"
                temp_path = os.path.join(os.getcwd(), temp_filename)
                tts.save(temp_path)
                with audio_lock:
                    pygame.mixer.init()
                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.delay(500)
                    pygame.mixer.quit()
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except PermissionError:
                        print("Could not delete temp file immediately.")
        except Exception as e:
            print(f"Error in speak function: {e}")

    threading.Thread(target=tts_worker).start()

# ---------- 2. Endpoint: supervised evaluation ----------
@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Reads labels.csv -> query,hospital_index,label
    Computes Precision / Recall / F1
    """
    df = pd.read_csv("labels.csv")   # file placed alongside app.py
    y_true = df["label"].values
    y_pred = np.ones_like(y_true)    # naive baseline: predict 1 for every record
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return jsonify({
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3)
    })


# ---------- 3. KNN recommender by city & specialization ----------
def knn_recommend(history, top_k=5):
    if not history:
        return []

    # correct keys
    X = np.array([
        [h.get("City", "").lower(), h.get("Specialization", "").lower()]
        for h in hospital_data
    ])

    city_enc = LabelEncoder().fit(X[:, 0])
    spec_enc = LabelEncoder().fit(X[:, 1])
    X_enc = np.column_stack([
        city_enc.transform(X[:, 0]),
        spec_enc.transform(X[:, 1])
    ])

    # most frequent city/spec from user history text
    tokens = " ".join(history).lower().split()
    city_counts = Counter([c for c in X[:, 0] if c in tokens])
    spec_counts = Counter([s for s in X[:, 1] if s in tokens])
    if not city_counts or not spec_counts:
        return []

    fav_city = city_counts.most_common(1)[0][0]
    fav_spec = spec_counts.most_common(1)[0][0]
    target = np.array([
        city_enc.transform([fav_city])[0],
        spec_enc.transform([fav_spec])[0]
    ]).reshape(1, -1)

    knn = KNeighborsClassifier(n_neighbors=min(top_k, len(X_enc)))
    knn.fit(X_enc, range(len(hospital_data)))
    idxs = knn.kneighbors(target, return_distance=False).flatten().tolist()

    return [hospital_data[i] for i in idxs[:top_k]]

@app.route("/recommend_knn_city_spec", methods=["POST"])
def recommend_knn_city_spec():
    """
    Body: { "history": ["query1","query2",...], "top_k": 5 }
    """
    data = request.get_json() or {}
    history = data.get("history", [])
    k = int(data.get("top_k", 5))
    recs = knn_recommend(history, top_k=k)
    return jsonify({"recommendations": recs})

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_query = data.get("query", "").strip()
    audio_file = data.get("audio")
    if audio_file:
        transcription = transcribe_audio(audio_file)
        user_query = transcription
    if not user_query:
        return jsonify({"response": "Please provide a valid query."})
    matching_hospitals = find_matching_hospitals(user_query)
    if matching_hospitals:
        response_json = jsonify({"response": matching_hospitals})
        threading.Thread(target=speak, args=(matching_hospitals,)).start()
        return response_json
    else:
        return jsonify({"response": "Sorry, no matching hospitals found."})

if __name__ == "__main__":
    app.run(debug=True)