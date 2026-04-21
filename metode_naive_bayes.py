import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F0F4F8; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1F4E79 0%, #2E75B6 100%);
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stSelectbox label { color: white !important; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #2E75B6;
        margin-bottom: 12px;
    }
    .metric-card.green  { border-left-color: #70AD47; }
    .metric-card.yellow { border-left-color: #FFC000; }
    .metric-card.red    { border-left-color: #FF0000; }
    .metric-card.blue   { border-left-color: #2E75B6; }
    .metric-val  { font-size: 2rem; font-weight: 700; color: #1F4E79; }
    .metric-label{ font-size: 0.85rem; color: #666; margin-top: 4px; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1F4E79, #2E75B6);
        color: white;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 1.05rem;
        font-weight: 600;
        margin: 20px 0 14px 0;
    }

    /* Result badge */
    .badge-layak       { background:#70AD47; color:white; padding:6px 16px; border-radius:20px; font-weight:700; font-size:1rem; }
    .badge-kurang      { background:#FFC000; color:white; padding:6px 16px; border-radius:20px; font-weight:700; font-size:1rem; }
    .badge-tidak       { background:#FF0000; color:white; padding:6px 16px; border-radius:20px; font-weight:700; font-size:1rem; }
    .badge-unknown     { background:#A0A0A0; color:white; padding:6px 16px; border-radius:20px; font-weight:700; font-size:1rem; }

    /* Divider */
    hr { border: 1px solid #D0D7DE; margin: 18px 0; }

    /* Dataframe */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Hide default streamlit header */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
LABEL_COL   = "LABEL_KONDISI"
FITUR_COLS  = ["TAHUN_PENGADAAN", "FREKUENSI_PEMAKAIAN", "UMUR_BARANG",
               "KONDISI_FISIK", "KELENGKAPAN"]
INFO_COLS   = ["NAMA_BARANG", "MERK", "KODE_BARANG", "KATEGORI_BARANG"]

COLOR_MAP = {
    "LAYAK":        "#70AD47",
    "KURANG LAYAK": "#FFC000",
    "TIDAK LAYAK":  "#FF0000",
}

def badge(label):
    cls = {"LAYAK":"badge-layak","KURANG LAYAK":"badge-kurang","TIDAK LAYAK":"badge-tidak"}.get(label,"badge-unknown")
    return f'<span class="{cls}">{label}</span>'

def metric_card(val, label, color="blue"):
    return f"""
    <div class="metric-card {color}">
        <div class="metric-val">{val}</div>
        <div class="metric-label">{label}</div>
    </div>"""

def section(title):
    st.markdown(f'<div class="section-header">📌 {title}</div>', unsafe_allow_html=True)

@st.cache_data
def load_csv_data():
    """Load data dari file CSV yang sudah ada"""
    try:
        # Ganti path sesuai lokasi file CSV Anda
        csv_path = "Data Inventaris SMK Muhammadiyah 12 - Tahun 2025.csv"
        if os.path.exists(csv_path):
            df_raw = pd.read_csv(csv_path)
            return df_raw
        else:
            st.error(f"File CSV tidak ditemukan: {csv_path}")
            return None
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

@st.cache_data
def load_and_train(df_raw):
    """Proses training model dari data CSV"""
    # ── Normalise column names ──────────────────────────────
    df_raw.columns = [c.strip().upper().replace(" ","_") for c in df_raw.columns]

    # Flexible column mapping
    rename = {}
    for c in df_raw.columns:
        if "NAMA" in c:        rename[c] = "NAMA_BARANG"
        elif "MERK" in c:      rename[c] = "MERK"
        elif "KODE" in c:      rename[c] = "KODE_BARANG"
        elif "KATEGORI" in c:  rename[c] = "KATEGORI_BARANG"
        elif "TAHUN" in c:     rename[c] = "TAHUN_PENGADAAN"
        elif "FREKUENSI" in c: rename[c] = "FREKUENSI_PEMAKAIAN"
        elif "UMUR" in c:      rename[c] = "UMUR_BARANG"
        elif "KONDISI_FISIK" in c: rename[c] = "KONDISI_FISIK"
        elif "KELENGKAPAN" in c: rename[c] = "KELENGKAPAN"
        elif "LABEL" in c:     rename[c] = "LABEL_KONDISI"
    df_raw.rename(columns=rename, inplace=True)

    df = df_raw.copy()

    # ── Preprocessing ──────────────────────────────────────
    df[FITUR_COLS] = df[FITUR_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    df_train = df[df[LABEL_COL].notna() & (df[LABEL_COL] != "TIDAK DIKETAHUI")].copy()
    df_train[LABEL_COL] = df_train[LABEL_COL].str.strip().str.upper()

    le = LabelEncoder()
    df_train["LABEL_ENC"] = le.fit_transform(df_train[LABEL_COL])

    X = df_train[FITUR_COLS]
    y = df_train["LABEL_ENC"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    gnb = GaussianNB();  gnb.fit(X_train, y_train)
    bnb = BernoulliNB(); bnb.fit(X_train, y_train)

    y_pred_gnb = gnb.predict(X_test)
    y_pred_bnb = bnb.predict(X_test)

    return df, df_train, le, gnb, bnb, X_train, X_test, y_train, y_test, y_pred_gnb, y_pred_bnb

@st.cache_data
def load_notebook():
    """Load dan tampilkan isi notebook"""
    try:
        if os.path.exists("naive-bayes.ipynb"):
            with open("naive-bayes.ipynb", "r", encoding="utf-8") as f:
                notebook = json.load(f)
            return notebook
        return None
    except:
        return None

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 Inventaris NB")
    st.markdown("### SMK Muhammadiyah 12")
    st.markdown("**Tahun 2025**")
    st.markdown("---")
    
    st.success("✅ Data otomatis dimuat!")
    st.info("📊 naive-bayes.ipynb")
    st.info("📋 Data Inventaris CSV")
    
    st.markdown("---")
    st.markdown("### 🔍 Navigasi")
    menu = st.radio("Pilih halaman:", [
        "🏠 Dashboard",
        "📋 Data Inventaris",
        "📓 Notebook Jupyter",
        "🤖 Model & Evaluasi",
        "📊 Confusion Matrix",
        "🔮 Prediksi Manual",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### ℹ️ Info")
    st.markdown("""
    **Algoritma:**  
    Gaussian NB & Bernoulli NB  
    
    **Split Data:**  
    80% Training / 20% Testing  
    
    **Referensi:**  
    Jurnal Iskandar Madani (2025)
    """)

# ─────────────────────────────────────────────────────────────
# MAIN - Load Data Otomatis
# ─────────────────────────────────────────────────────────────
@st.cache_data
def initialize_app():
    """Inisialisasi aplikasi dengan load data otomatis"""
    # Load notebook
    notebook = load_notebook()
    
    # Load CSV
    df_raw = load_csv_data()
    
    if df_raw is None or notebook is None:
        return None, None, None
    
    # Train model
    result = load_and_train(df_raw)
    return notebook, df_raw, result

notebook, df_raw, model_result = initialize_app()

if model_result is None:
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px;'>
        <h1 style='color:#1F4E79;'>📦 Sistem Klasifikasi Inventaris</h1>
        <h3 style='color:#2E75B6;'>SMK Muhammadiyah 12 - Tahun 2025</h3>
        <p style='color:#666; font-size:1.1rem;'>
            Pastikan file berikut ada di folder yang sama:<br>
            - <code>naive-bayes.ipynb</code><br>
            - <code>Data Inventaris SMK Muhammadiyah 12 - Tahun 2025.csv</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Unpack model results
(df, df_train, le, gnb, bnb,
 X_train, X_test, y_train, y_test,
 y_pred_gnb, y_pred_bnb) = model_result

label_counts = df_train[LABEL_COL].value_counts()
best_model   = gnb

# ─────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════
if menu == "🏠 Dashboard":
    st.markdown("<h1 style='color:#1F4E79;'>🏠 Dashboard Inventaris SMK Muhammadiyah 12</h1>", unsafe_allow_html=True)
    st.markdown("**Tahun 2025** - Analisis kondisi barang inventaris menggunakan Naive Bayes.")
    st.markdown("---")

    # ── Metric cards ───────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card(len(df), "Total Barang", "blue"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(label_counts.get("LAYAK", 0), "Layak", "green"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(label_counts.get("KURANG LAYAK", 0), "Kurang Layak", "yellow"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card(label_counts.get("TIDAK LAYAK", 0), "Tidak Layak", "red"), unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        section("Distribusi Label Kondisi Barang")
        fig, ax = plt.subplots(figsize=(7, 4))
        labels  = label_counts.index.tolist()
        values  = label_counts.values.tolist()
        colors  = [COLOR_MAP.get(l, "#999") for l in labels]
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.set_title("Distribusi Barang Berdasarkan Label Kondisi", fontweight='bold', fontsize=12)
        ax.set_xlabel("Label Kondisi"); ax.set_ylabel("Jumlah Barang")
        ax.set_facecolor("#F8F9FA"); fig.patch.set_facecolor("white")
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig, use_container_width=True)

    with col_right:
        section("Proporsi Kondisi (%)")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        wedge_colors = [COLOR_MAP.get(l, "#999") for l in labels]
        wedges, texts, autotexts = ax2.pie(
            values, labels=labels, colors=wedge_colors,
            autopct='%1.1f%%', startangle=140,
            wedgeprops=dict(edgecolor='white', linewidth=2)
        )
        for at in autotexts:
            at.set_fontsize(11); at.set_fontweight('bold'); at.set_color('white')
        ax2.set_title("Proporsi Kondisi", fontweight='bold', fontsize=12)
        st.pyplot(fig2, use_container_width=True)

    st.markdown("---")

    section("Akurasi Model")
    m1, m2 = st.columns(2)
    acc_g = accuracy_score(y_test, y_pred_gnb) * 100
    acc_b = accuracy_score(y_test, y_pred_bnb) * 100
    with m1:
        st.markdown(metric_card(f"{acc_g:.2f}%", "Gaussian NB", "blue"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card(f"{acc_b:.2f}%", "Bernoulli NB", "blue"), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: DATA INVENTARIS
# ═══════════════════════════════════════════════════════════
elif menu == "📋 Data Inventaris":
    st.markdown("<h1 style='color:#1F4E79;'>📋 Data Inventaris Lengkap</h1>", unsafe_allow_html=True)
    st.markdown(f"**SMK Muhammadiyah 12 - Tahun 2025** | Total: {len(df)} data")
    st.markdown("---")

    # Filter
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_label = st.selectbox("Filter Label:", ["Semua"] + sorted(df_train[LABEL_COL].unique().tolist()))
    with col_f2:
        filter_kat = st.selectbox("Filter Kategori:", ["Semua"] + (sorted(
            df_train["KATEGORI_BARANG"].unique().tolist()) if "KATEGORI_BARANG" in df_train.columns else []))
    with col_f3:
        search = st.text_input("🔍 Cari nama barang:", "")

    df_view = df_train.copy()
    if filter_label != "Semua":
        df_view = df_view[df_view[LABEL_COL] == filter_label]
    if filter_kat != "Semua" and "KATEGORI_BARANG" in df_view.columns:
        df_view = df_view[df_view["KATEGORI_BARANG"] == filter_kat]
    if search:
        df_view = df_view[df_view["NAMA_BARANG"].str.contains(search, case=False, na=False)]

    st.markdown(f"**Menampilkan {len(df_view)} dari {len(df_train)} data**")

    show_cols = [c for c in INFO_COLS + FITUR_COLS + [LABEL_COL] if c in df_view.columns]

    def color_label(val):
        colors = {"LAYAK":"background-color:#C6EFCE;color:#276221",
                  "KURANG LAYAK":"background-color:#FFEB9C;color:#7F6000",
                  "TIDAK LAYAK":"background-color:#FFC7CE;color:#9C0006"}
        return colors.get(val, "")

    styled = df_view[show_cols].reset_index(drop=True).style.applymap(
        color_label, subset=[LABEL_COL])
    st.dataframe(styled, use_container_width=True, height=600)

    section("Statistik Deskriptif")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_train[FITUR_COLS].describe().round(2), use_container_width=True)
    with col2:
        st.dataframe(df_train[LABEL_COL].value_counts().reset_index(), use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: NOTEBOOK JUPYTER
# ═══════════════════════════════════════════════════════════
elif menu == "📓 Notebook Jupyter":
    st.markdown("<h1 style='color:#1F4E79;'>📓 naive-bayes.ipynb</h1>", unsafe_allow_html=True)
    st.markdown("**Isi notebook lengkap yang digunakan untuk analisis dan training model.**")
    st.markdown("---")