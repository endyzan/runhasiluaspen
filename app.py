import streamlit as st
import pickle
import numpy as np
import sklearn


# Memuat model yang telah disimpan dengan exception handling
model = None
model_loading_error = ""

try:
    with open('bagging_classifier.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    model_loading_error = str(e)
    st.error(f"Error loading the model: {model_loading_error}")

# Judul aplikasi
st.title("Bagging Classifier Prediction")

if model is not None:
    st.write("Model loaded successfully!")

    # Form input data
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=0, max_value=100, step=1)
        sex = st.number_input("Sex", min_value=0, max_value=1, step=1)
        steroid = st.number_input("Steroid", min_value=0, max_value=1, step=1)
        antivirals = st.number_input("Antivirals", min_value=0, max_value=1, step=1)
        fatigue = st.number_input("Fatigue", min_value=0, max_value=1, step=1)
        malaise = st.number_input("Malaise", min_value=0, max_value=1, step=1)
        anorexia = st.number_input("Anorexia", min_value=0, max_value=1, step=1)
        liver_big = st.number_input("Liver Big", min_value=0, max_value=1, step=1)
        liver_firm = st.number_input("Liver Firm", min_value=0, max_value=1, step=1)
        spleen_palpable = st.number_input("Spleen Palpable", min_value=0, max_value=1, step=1)
        spiders = st.number_input("Spiders", min_value=0, max_value=1, step=1)
        ascites = st.number_input("Ascites", min_value=0, max_value=1, step=1)
        varices = st.number_input("Varices", min_value=0, max_value=1, step=1)
        bilirubin = st.number_input("Bilirubin", min_value=0, max_value=100, step=1)
        alk_phosphate = st.number_input("Alk Phosphate", min_value=0, max_value=100, step=1)
        sgot = st.number_input("Sgot", min_value=0, max_value=100, step=1)
        albumin = st.number_input("Albumin", min_value=0, max_value=100, step=1)
        protime = st.number_input("Protime", min_value=0, max_value=100, step=1)
        histology = st.number_input("Histology", min_value=0, max_value=1, step=1)
        submit_button = st.form_submit_button("Predict")

    if submit_button:
        features = [age, sex, steroid, antivirals, fatigue, malaise, anorexia, liver_big, liver_firm,
                    spleen_palpable, spiders, ascites, varices, bilirubin, alk_phosphate, sgot, albumin,
                    protime, histology]
        final_features = [np.array(features)]

        # Melakukan prediksi menggunakan model yang telah dilatih
        predictions = [estimator.predict(final_features)[0] for estimator in model['estimators']]

        # Majority vote untuk prediksi akhir
        pred_majority_vote = np.bincount(predictions).argmax()

        # Menginterpretasi hasil prediksi
        output = 'Class 1' if pred_majority_vote == 1 else 'Class 2'

        # Menampilkan hasil prediksi
        st.write(f"Predicted Class: {output}")
        st.write(f"Input Features: {features}")
        st.write(f"Model Accuracy: 0.97")  # Sesuaikan dengan akurasi model Anda
else:
    st.error(f"Model loading failed: {model_loading_error}")
