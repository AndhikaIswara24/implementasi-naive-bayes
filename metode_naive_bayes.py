import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")
import json
import os

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inventaris Naive Bayes - SMK Muhammadiyah 12",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# FIXED CSS - FONT HITAM SEMUA
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ===== GLOBAL TEXT FIX ===== */
html, body, [class*="css"]  {
    color: #000000 !important;
}

/* Main App */
.stApp {
    background-color: #F0F4F8;
    color: #000000 !important;
}

/* Semua teks utama */
h1, h2, h3, h4, h5, h6,
p, span, div, label,
strong, small {
    color: #000000 !important;
    opacity: 1 !important;
}

/* Judul Streamlit */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 {
    color: #000000 !important;
}

/* Markdown */
.stMarkdown {
    color: #000000 !important;
}

/* Sidebar tetap putih */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1F4E79 0%, #2E75B6 100%);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Hilangkan transparan */
* {
    opacity: 1 !important;
}

/* Hide menu */
#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
LABEL_COL = "Label Kondisi"
FITUR_COLS = ["TAHUN_PENGADAAN", "FREKUENSI_PEMAKAIAN", "UMUR_BARANG", "KONDISI_FISIK", "KELENGKAPAN"]
INFO_COLS = ["NAMA_BARANG", "MERK", "KODE_BARANG", "KATEGORI_BARANG"]

COLOR_MAP = {"LAYAK": "#70AD47", "KURANG LAYAK": "#FFC000", "TIDAK LAYAK": "#FF0000"}

def badge(label):
    cls = {"LAYAK":"badge-layak","KURANG LAYAK":"badge-kurang","TIDAK LAYAK":"badge-tidak"}.get(label,"")
    return f'<span class="{cls}">{label}</span>'

def metric_card(val, label, color="blue"):
    return f'''
    <div class="metric-card {color}">
        <div class="metric-val">{val}</div>
        <div class="metric-label">{label}</div>
    </div>
    '''

def section(title):
    st.markdown(f'<div class="section-header">📌 {title}</div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    csv_path = "Data Inventaris SMK Muhammadiyah 12 - Tahun 2025.csv"
    try:
        if not os.path.exists(csv_path):
            st.error(f"❌ **File CSV TIDAK DITEMUKAN**: {csv_path}")
            st.stop()
        
        # Try header=0 first, then header=1
        try:
            df_raw = pd.read_csv(csv_path)
        except:
            df_raw = pd.read_csv(csv_path, header=1)
        
        st.sidebar.success(f"✅ Loaded: **{len(df_raw):,d}** rows")
        return df_raw
    except Exception as e:
        st.error(f"❌ **Error loading CSV**: {str(e)}")
        st.stop()

@st.cache_data
def preprocess_and_train(df_raw):
    # Normalize columns
    df_raw = df_raw.copy()
    df_raw.columns = [str(c).strip().upper().replace(" ", "_") for c in df_raw.columns]

    # ── Comprehensive column mapping ──────────────────────────
    rename_map = {}
    for c in df_raw.columns:
        cu = c.upper()
        if   "NAMA"        in cu and "BARANG" in cu: rename_map[c] = "NAMA_BARANG"
        elif "MERK"        in cu:                    rename_map[c] = "MERK"
        elif "KODE"        in cu:                    rename_map[c] = "KODE_BARANG"
        elif "KATEGORI"    in cu:                    rename_map[c] = "KATEGORI_BARANG"
        elif "TAHUN"       in cu:                    rename_map[c] = "TAHUN_PENGADAAN"
        elif "FREKUENSI"   in cu:                    rename_map[c] = "FREKUENSI_PEMAKAIAN"
        elif "UMUR"        in cu:                    rename_map[c] = "UMUR_BARANG"
        elif "FISIK"       in cu:                    rename_map[c] = "KONDISI_FISIK"   # ← khusus fisik dulu
        elif "KELENGKAPAN" in cu:                    rename_map[c] = "KELENGKAPAN"
        # FIX: tangkap lebih banyak variasi nama kolom label
        elif any(x in cu for x in ["LABEL", "STATUS", "KETERANGAN", "KLASIFIKASI"]):
            rename_map[c] = "Label Kondisi"

    df_raw.rename(columns=rename_map, inplace=True)

    # ── FIX: Auto-detect kolom label dari ISI DATA ─────────────
    if "Label Kondisi" not in df_raw.columns:
        KNOWN_LABELS = {"LAYAK", "KURANG LAYAK", "TIDAK LAYAK",
                        "KURANG_LAYAK", "TIDAK_LAYAK"}
        for col in df_raw.columns:
            # Lewati kolom yang sudah di-assign
            if col in (list(rename_map.values()) + ["NAMA_BARANG", "MERK",
               "KODE_BARANG", "KATEGORI_BARANG", "TAHUN_PENGADAAN",
               "FREKUENSI_PEMAKAIAN", "UMUR_BARANG", "KONDISI_FISIK", "KELENGKAPAN"]):
                continue
            unique_vals = set(
                df_raw[col].dropna().astype(str)
                .str.strip().str.upper().unique()
            )
            if unique_vals & KNOWN_LABELS:          # ada irisan → ini kolom label
                df_raw.rename(columns={col: "Label Kondisi"}, inplace=True)
                st.sidebar.info(f"🔍 Label kolom: **{col}** → 'Label Kondisi'")
                break

    # ── Gagal total → tampilkan kolom tersedia untuk debug ────
    if "Label Kondisi" not in df_raw.columns:
        st.error("❌ Kolom 'Label Kondisi' tidak ditemukan!")
        st.warning("📋 Kolom yang tersedia di CSV:")
        st.code("\n".join(df_raw.columns.tolist()))
        st.info("💡 Pastikan salah satu kolom berisi nilai: LAYAK / KURANG LAYAK / TIDAK LAYAK")
        st.stop()

    df = df_raw.copy()

    # Numeric features
    available_features = [col for col in FITUR_COLS if col in df.columns]
    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Training data
    df_train = df[df["Label Kondisi"].notna()].copy()
    df_train["Label Kondisi"] = (
        df_train["Label Kondisi"].astype(str).str.strip().str.upper()
    )

    le = LabelEncoder()
    df_train["LABEL_ENC"] = le.fit_transform(df_train["Label Kondisi"])

    X = df_train[available_features]
    y = df_train["LABEL_ENC"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(y.unique()) > 1 else None
    )

    gnb = GaussianNB()
    bnb = BernoulliNB()
    gnb.fit(X_train, y_train)
    bnb.fit(X_train, y_train)

    y_pred_gnb = gnb.predict(X_test)
    y_pred_bnb = bnb.predict(X_test)

    return (df, df_train, le, gnb, bnb,
            X_train, X_test, y_train, y_test,
            y_pred_gnb, y_pred_bnb, available_features)

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
try:
    st.title("📦 **Sistem Inventaris Naive Bayes**")
    st.markdown("***SMK Muhammadiyah 12 - Tahun 2025***")
    
    df_raw = load_data()
    model_data = preprocess_and_train(df_raw)
    
    (df, df_train, le, gnb, bnb, X_train, X_test, y_train, y_test, y_pred_gnb, y_pred_bnb, feature_cols) = model_data
    label_counts = df_train["Label Kondisi"].value_counts()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📦 **Inventaris NB**")
        st.markdown("**SMK Muhammadiyah 12**")
        st.metric("Total Barang", len(df))
        st.metric("Data Training", len(df_train))
        st.markdown("---")
        
        menu = st.radio("📂 Menu:", [
            "🏠 Dashboard",
            "📋 Data Inventaris", 
            "🤖 Model Evaluasi", 
            "📊 Confusion Matrix",
            "🔮 Prediksi Baru"
        ])
    
    # Pages
    if menu == "🏠 Dashboard":
        st.markdown("### 🏠 **Dashboard Utama**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(metric_card(f"{len(df):,d}", "Total Barang", "blue"), unsafe_allow_html=True)
        with col2: st.markdown(metric_card(label_counts.get("LAYAK", 0), "Layak ✅", "green"), unsafe_allow_html=True)
        with col3: st.markdown(metric_card(label_counts.get("KURANG LAYAK", 0), "Kurang ⚠️", "yellow"), unsafe_allow_html=True)
        with col4: st.markdown(metric_card(label_counts.get("TIDAK LAYAK", 0), "Tidak Layak ❌", "red"), unsafe_allow_html=True)
        
        col_left, col_right = st.columns([2, 1])
        with col_left:
            section("📊 Distribusi Kondisi")
            fig, ax = plt.subplots(figsize=(9, 5))
            colors = [COLOR_MAP.get(l, "#ccc") for l in label_counts.index]
            bars = ax.bar(label_counts.index, label_counts.values, color=colors, edgecolor='white', linewidth=2)
            ax.set_title("Distribusi Barang Inventaris", fontweight='bold', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col_right:
            section("🎯 Akurasi Model")
            acc_gnb = accuracy_score(y_test, y_pred_gnb) * 100
            acc_bnb = accuracy_score(y_test, y_pred_bnb) * 100
            st.markdown(metric_card(f"{acc_gnb:.1f}%", "Gaussian NB", "blue"), unsafe_allow_html=True)
            st.markdown(metric_card(f"{acc_bnb:.1f}%", "Bernoulli NB", "blue"), unsafe_allow_html=True)
    
    elif menu == "📋 Data Inventaris":
        st.markdown("### 📋 **Data Inventaris Lengkap**")
        
        col1, col2 = st.columns(2)
        with col1:
            filter_label = st.selectbox("Filter Kondisi:", ["Semua"] + sorted(df_train["Label Kondisi"].unique().tolist()))
        with col2:
            search = st.text_input("🔍 Cari nama barang:", placeholder="Ketik nama barang...")
        
        df_view = df_train.copy()
        if filter_label != "Semua":
            df_view = df_view[df_view["Label Kondisi"] == filter_label]
        if search:
            df_view = df_view[df_view["NAMA_BARANG"].str.contains(search, case=False, na=False)]
        
        st.info(f"**Menampilkan {len(df_view):,d} dari {len(df_train):,d} data**")
        st.dataframe(df_view, use_container_width=True, height=600)
    
    elif menu == "🤖 Model Evaluasi":
        st.markdown("### 🤖 **Evaluasi Model Naive Bayes**")
        
        col1, col2 = st.columns(2)
        with col1:
            section("Gaussian Naive Bayes")
            acc_g = accuracy_score(y_test, y_pred_gnb)
            st.metric("Accuracy", f"{acc_g*100:.2f}%")
        
        with col2:
            section("Bernoulli Naive Bayes")
            acc_b = accuracy_score(y_test, y_pred_bnb)
            st.metric("Accuracy", f"{acc_b*100:.2f}%")
    
    elif menu == "📊 Confusion Matrix":
        st.markdown("### 📊 **Confusion Matrix**")
        model_choice = st.selectbox("Pilih Model:", ["Gaussian NB", "Bernoulli NB"])
        y_pred = y_pred_gnb if model_choice == "Gaussian NB" else y_pred_bnb
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_title(f"Confusion Matrix - {model_choice}", fontsize=14, fontweight='bold')
        st.pyplot(fig)
    
    elif menu == "🔮 Prediksi Baru":
        st.markdown("### 🔮 **Prediksi Kondisi Barang Baru**")
        
        col_form, col_result = st.columns([1, 1])
        
        with col_form:
            section("📝 Input Data")
            nama_barang = st.text_input("Nama Barang:", placeholder="Contoh: Laptop Dell")
            model_pilihan = st.selectbox("Model:", ["Gaussian NB", "Bernoulli NB"])
            
            st.markdown("**Fitur Numerik:**")
            tahun = st.slider("📅 Tahun Pengadaan", 2015, 2025, 2023)
            frekuensi = st.slider("⏰ Frekuensi Pemakaian (1=Jarang, 5=Sering)", 1, 5, 3)
            umur = st.slider("📏 Umur Barang (tahun)", 0, 10, 2)
            kondisi_fisik = st.slider("🔧 Kondisi Fisik (1=Rusak, 5=Baik)", 1, 5, 4)
            kelengkapan = st.slider("📦 Kelengkapan (1=Tidak lengkap, 5=Lengkap)", 1, 5, 4)
            
            if st.button("🔮 **Lakukan Prediksi**", use_container_width=True, type="primary"):
                # Predict
                input_data = np.array([[tahun, frekuensi, umur, kondisi_fisik, kelengkapan]])
                model = gnb if model_pilihan == "Gaussian NB" else bnb
                prediction = model.predict(input_data)[0]
                pred_label = le.inverse_transform([prediction])[0]
                
                with col_result:
                    section("📊 Hasil Prediksi")
                    st.markdown(f"**Barang:** {nama_barang or 'Tidak disebutkan'}")
                    st.markdown(f"**Prediksi:** {badge(pred_label)}", unsafe_allow_html=True)
                    
                    st.success(f"Model: **{model_pilihan}**")
                    st.info(f"Input: Tahun={tahun}, Frek={frekuensi}, Umur={umur}, Fisik={kondisi_fisik}, Lengkap={kelengkapan}")

except Exception as e:
    st.error(f"❌ **Terjadi Error**: {str(e)}")
    st.info("Pastikan file CSV ada di folder yang sama dengan app.py")