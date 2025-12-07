# ======================================================
# STREAMLIT APPLICATION — FINAL FIXED VERSION
# Feature name mismatch resolved: age + duration
# ======================================================

import streamlit as st
import json
import joblib
import os
import numpy as np
import pandas as pd
import networkx as nx
import re

# ======================================================
# LOAD ARTIFACTS AND MODEL
# ======================================================

BASE_DIR = "/content/drive/MyDrive/Colab Notebooks/MILESTONE 2-3/Model Notebooks/"
ARTIFACT_DIR = BASE_DIR + "artifacts/"
MODEL_DIR = BASE_DIR + "model_artifacts/"

st.write("Loading artifacts from:", ARTIFACT_DIR)
st.write("Loading model from:", MODEL_DIR)

with open(ARTIFACT_DIR + "symptom_map.json", "r", encoding="utf-8") as f:
    symptom_map = json.load(f)

with open(ARTIFACT_DIR + "keyword_lookup.json", "r", encoding="utf-8") as f:
    keyword_lookup = json.load(f)

rf_model = joblib.load(MODEL_DIR + "child_disease_random_forest.pkl")

# ======================================================
# NLP FUNCTIONS (FIXED METADATA NAMES)
# ======================================================

negation_words = {"no", "not", "without", "tet", "si", "siri", "sita", "siko"}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u0100-\uffff\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def is_negated(tokens, idx, window=3):
    start = max(0, idx - window)
    end = min(len(tokens), idx + window + 1)
    return any(tokens[i] in negation_words for i in range(start, end))

def extract_symptoms(text):
    cleaned = clean_text(text)
    tokens = cleaned.split()

    detected = {sym: False for sym in symptom_map.keys()}

    for kw, symptom in keyword_lookup.items():
        if kw in cleaned:
            idx = cleaned.find(kw)
            token_index = len(cleaned[:idx].split())
            if not is_negated(tokens, token_index):
                detected[symptom] = True

    # FIX: use "age" and "duration" to match the model
    age = 2
    duration = 1

    if m := re.search(r"(\d+)\s*year", cleaned):
        age = int(m.group(1))

    if m := re.search(r"(\d+)\s*(day|days|d)", cleaned):
        duration = int(m.group(1))

    return detected, {"age": age, "duration": duration}

# ======================================================
# KNOWLEDGE GRAPH RULES
# ======================================================

KG = nx.DiGraph()

KG.add_edge("fast_breathing", "pneumonia", weight=0.9)
KG.add_edge("cough", "pneumonia", weight=0.7)
KG.add_edge("fever", "pneumonia", weight=0.4)

KG.add_edge("fever", "malaria", weight=0.8)
KG.add_edge("vomiting", "malaria", weight=0.6)
KG.add_edge("weakness", "malaria", weight=0.5)

KG.add_edge("diarrhea", "diarrhea", weight=0.9)
KG.add_edge("vomiting", "diarrhea", weight=0.5)
KG.add_edge("fever", "diarrhea", weight=0.3)

diseases = ["pneumonia", "malaria", "diarrhea"]

danger_signs = [
    "convulsions", "chest_indrawing", "unable_to_feed",
    "vomiting_everything", "lethargic"
]

def kg_reasoning(symptoms):
    scores = {d: 0 for d in diseases}
    explanations = []
    danger = []

    for s, present in symptoms.items():
        if present and s in KG.nodes:
            for _, disease, data in KG.out_edges(s, data=True):
                scores[disease] += data["weight"]
                explanations.append(f"{s} → {disease} (weight {data['weight']})")

    for ds in danger_signs:
        if symptoms.get(ds, False):
            danger.append(ds)

    risk = "high" if danger else "moderate" if max(scores.values()) >= 0.7 else "low"
    return scores, danger, risk, explanations

# ======================================================
# HYBRID REASONING ENGINE (FIXED feature names)
# ======================================================

def hybrid_reasoning(symptoms, metadata):

    fv = [
        int(symptoms["fever"]),
        int(symptoms["cough"]),
        int(symptoms["fast_breathing"]),
        int(symptoms["diarrhea"]),
        int(symptoms["vomiting"]),
        int(symptoms["weakness"]),
        int(symptoms["poor_feeding"]),
        int(symptoms["convulsions"]),
        metadata["age"],       # ✔ FIXED
        metadata["duration"]   # ✔ FIXED
    ]

    df = pd.DataFrame([fv], columns=[
        "fever", "cough", "fast_breathing", "diarrhea", "vomiting",
        "weakness", "poor_feeding", "convulsions",
        "age", "duration"     # ✔ FIXED COLUMN NAMES
    ])

    probs = rf_model.predict_proba(df)[0]
    clf_pred = rf_model.predict(df)[0]
    clf_probs = {label: float(p) for label, p in zip(rf_model.classes_, probs)}

    kg_scores, danger, risk, kg_rules = kg_reasoning(symptoms)

    final_scores = {d: 0.7*clf_probs[d] + 0.3*kg_scores[d] for d in diseases}
    final_pred = max(final_scores, key=final_scores.get)

    return final_pred, risk, clf_probs, kg_scores, kg_rules, danger

# ======================================================
# STREAMLIT UI
# ======================================================

st.title("🩺 VHT Childhood Disease Diagnosis Assistant")
st.write("Enter symptoms in Luganda or English, or select a test case from the sidebar.")

st.sidebar.header("🧪 Test Cases")

test_cases = {
    "Select a test case": "",
    "Pneumonia (Fast Breathing)": "Omwana afuuya mangu era akyawa okukosora.",
    "Malaria (Fever + Weakness)": "Omwana alina omusujja era alina obunafu.",
    "Diarrhea Case": "Omwana alina endwadde y'ekidugavu era ali kusgala.",
    "Danger Signs (Convulsions)": "Omwana alina obutonya obungi era tayinza kulya.",
    "English Case (Malaria-like)": "The child has had a high fever for 3 days and is vomiting.",
    "English Case (Pneumonia-like)": "The child is breathing very fast and coughing."
}

selected_case = st.sidebar.selectbox("Choose Test Case:", list(test_cases.keys()))
default_text = test_cases[selected_case]

input_text = st.text_area("Symptoms:", value=default_text, height=150)

if st.button("Analyze"):

    if not input_text.strip():
        st.warning("⚠️ Please enter symptoms or choose a test case.")
        st.stop()

    symptoms, metadata = extract_symptoms(input_text)
    final_pred, risk, clf_probs, kg_scores, kg_rules, danger = hybrid_reasoning(symptoms, metadata)

    st.subheader("Final Diagnosis")
    st.success(final_pred.upper())

    st.subheader("Risk Level")
    st.warning(risk.upper())

    st.subheader("Detected Symptoms")
    st.json(symptoms)

    st.subheader("ML Probabilities")
    st.json(clf_probs)

    st.subheader("Knowledge Graph Rules Fired")
    for rule in kg_rules:
        st.write("•", rule)

    if danger:
        st.error(f"🚨 DANGER SIGNS DETECTED: {', '.join(danger)}")
