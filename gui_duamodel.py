import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# -----------------------------
# Set Page
# -----------------------------
st.set_page_config(page_title="Klasifikasi Varietas Sawi", page_icon="🥬", layout="centered")

# -----------------------------
# Load Models
# -----------------------------
model1_path = "0.0001FIXINIskripsikamila.h5"
model2_path = "BISMILLAHYAALLAHINIGUI_skripsikamila.h5"
model1 = load_model(model1_path) if os.path.exists(model1_path) else None
model2 = load_model(model2_path) if os.path.exists(model2_path) else None

# -----------------------------
# Class Names
# -----------------------------
class_names = {
    "sawi_caisim": "Sawi Caisim",
    "sawi_kailan": "Sawi Kailan",
    "sawi_pahit": "Sawi Pahit",
    "sawi_pakcoy": "Sawi Pakcoy",
    "sawi_putih": "Sawi Putih",
    "unknown": "BUKAN SAWI!"}

# -----------------------------
# Info Varietas
# -----------------------------
varietas_info = {
    "Sawi Caisim": {"Karakteristik": "Daun lebar, tipis, dan halus. Warna hijau merata. Batang ramping dan memanjang."},
    "Sawi Kailan": {"Karakteristik": "Daun tebal dan mengilap. Warna hijau tua kebiruan. Batang besar dan tampak kokoh."},
    "Sawi Pahit": {"Karakteristik": "Daun keriting dan kasar. Warna hijau tua dengan urat daun yang jelas terlihat."},
    "Sawi Pakcoy": {"Karakteristik": "Daun hijau tua dengan batang putih tebal dan menggembung. Daun berdiri tegak dan berbentuk oval."},
    "Sawi Putih": {"Karakteristik": "Daun besar dan bergelombang. Warna putih kekuningan di batang dan tulang daun, ujung daun hijau pucat."},
    "BUKAN SAWI!": {"Karakteristik": "Tidak termasuk dalam 5 varietas sawi utama."}}

# Header
st.markdown("""
<h1 style='text-align: center;'>Klasifikasi Varietas Sawi</h1>
<h4 style='text-align: center;'>Informasi Karakteristik</h4>
""", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("Upload gambar sawi", type=["jpg", "jpeg"])

# Sebelum Upload: Tampilkan Semua Info Karakteristik
if not uploaded_file:
    st.markdown("### 📋 Informasi Semua Varietas")
    df_varietas = {
        "Nama Varietas": list(varietas_info.keys())[:-1],
        "Karakteristik": [v["Karakteristik"] for k, v in varietas_info.items() if k != "BUKAN SAWI!"]
    }
    df = pd.DataFrame(df_varietas)
    df.index = [''] * len(df)
    st.table(df)

# Setelah Upload
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    width, height = img.size
    max_dim = max(width, height)
    new_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2
    new_img.paste(img, (paste_x, paste_y))
    img_resized = new_img.resize((224, 224), Image.LANCZOS)

    col_main1, col_main2 = st.columns([4, 6])
    with col_main1:
        st.image(img_resized, width=270)

    with col_main2:
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        name2 = "BUKAN SAWI!"
        conf2 = 0

        if model2: # Prediksi Model 2
            pred2 = model2.predict(img_array)
            idx2 = np.argmax(pred2)
            key2 = list(class_names.keys())[idx2]
            name2 = class_names[key2]
            conf2 = np.max(pred2) * 100

            st.markdown(f"""
            <div style='padding: 1rem; background-color: #dff0d8; border-radius: 8px;'>
            <h3 style='margin: 0;'>Hasil Prediksi:</h3>
            <h2 style='margin: 0; color: #3c763d;'>✅ {name2}</h2>
            </div>
""", unsafe_allow_html=True)


        # Informasi karakteristik varietas hasil deteksi
        varietas_key = name2 if name2 in varietas_info else "BUKAN SAWI!"
        karakteristik = varietas_info[varietas_key]["Karakteristik"]

        st.markdown(f"""
        <div style='padding: 0.8rem; background-color: #eef5fa; border-radius: 6px;'>
        <p style='font-size: 14px; margin: 0 0 5px 0;'><strong>Karakteristik Varietas:</strong></p>
        <p style='font-size: 13px; margin: 0;'>{karakteristik}</p>
        </div>
        """, unsafe_allow_html=True)

        # Prediksi model 1
        if model1:
            pred1 = model1.predict(img_array)
            idx1 = np.argmax(pred1)
            key1 = list(class_names.keys())[idx1]
            name1 = class_names[key1]
            conf1 = np.max(pred1) * 100

            st.markdown(
                f"<p style='font-size: 13px; color: gray; margin-top:10px;'>Prediksi dengan Model 1: {name1}</p>",
                unsafe_allow_html=True)

# Cek model
if not model1 or not model2:
    st.error("❌ Salah satu model tidak ditemukan. Pastikan file .h5 tersedia di folder.")