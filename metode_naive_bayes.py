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
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #F0F4F8; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1F4E79 0%, #2E75B6 100%);
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    .metric-card {
        background: white; border-radius: 12px; padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 5px solid #2E75B6;
        margin-bottom: 12px;
    }
    .metric-card.green  { border-left-color: #70AD47; }
    .metric-card.yellow { border-left-color: #FFC000; }
    .metric-card.red    { border-left-color: #FF0000; }
    .metric-card.blue   { border-left-color: #2E75B6; }
    .metric-val  { font-size: 2rem; font-weight: 700; color: #1F4E79; }
    .metric-label{ font-size: 0.85rem; color: #666; margin-top: 4px; }
    .section-header {
        background: linear-gradient(90deg, #1F4E79, #2E75B6); color: white;
        padding: 10px 18px; border-radius: 8px; font-size: 1.05rem;
        font-weight: 600; margin: 20px 0 14px 0;
    }
    .badge-layak    { background:#70AD47; color:white; padding:6px 16px; border-radius:20px; font-weight:700; }
    .badge-kurang   { background:#FFC000; color:white; padding:6px 16px; border-radius:20px; font-weight:700; }
    .badge-tidak    { background:#FF0000; color:white; padding:6px 16px; border-radius:20px; font-weight:700; }
    hr { border: 1px solid #D0D7DE; margin: 18px 0; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────
LABEL_COL = "LABEL_KONDISI"
FITUR_COLS = ["TAHUN_PENGADAAN", "FREKUENSI_PEMAKAIAN", "UMUR_BARANG", "KONDISI_FISIK", "KELENGKAPAN"]
INFO_COLS = ["NAMA_BARANG", "MERK", "KODE_BARANG", "KATEGORI_BARANG"]

COLOR_MAP = {"LAYAK": "#70AD47", "KURANG LAYAK": "#FFC000", "TIDAK LAYAK": "#FF0000"}

def badge(label):
    cls = {"LAYAK":"badge-layak","KURANG LAYAK":"badge-kurang","TIDAK LAYAK":"badge-tidak"}.get(label,"")
    return f'<span class="{cls}">{label}</span>'

def metric_card(val, label, color="blue"):
    return f'<div class="metric-card {color}"><div class="metric-val">{val}</div><div class="metric-label">{label}</div></div>'

def section(title):
    st.markdown(f'<div class="section-header">📌 {title}</div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load data CSV secara otomatis - HAPUS UPLOAD"""
    csv_path = "Data Inventaris SMK Muhammadiyah 12 - Tahun 2025.csv"
    try:
        if os.path.exists(csv_path):
            df_raw = pd.read_csv(csv_path)
            st.sidebar.success(f"✅ Loaded: {len(df_raw)} rows")
            return df_raw
        else:
            st.error(f"❌ File tidak ditemukan: {csv_path}")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        st.stop()

@st.cache_data
def preprocess_and_train(df_raw):
    """Preprocessing dan training model"""
    df_raw.columns = [c.strip().upper().replace(" ","_") for c in df_raw.columns]
    
    # Column mapping
    rename_map = {}
    for c in df_raw.columns:
        if any(x in c for x in ["NAMA", "NAMA BARANG"]): rename_map[c] = "NAMA_BARANG"
        elif "MERK" in c: rename_map[c] = "MERK"
        elif "KODE" in c: rename_map[c] = "KODE_BARANG"
        elif "KATEGORI" in c: rename_map[c] = "KATEGORI_BARANG"
        elif "TAHUN" in c: rename_map[c] = "TAHUN_PENGADAAN"
        elif "FREKUENSI" in c: rename_map[c] = "FREKUENSI_PEMAKAIAN"
        elif "UMUR" in c: rename_map[c] = "UMUR_BARANG"
        elif "KONDISI_FISIK" in c: rename_map[c] = "KONDISI_FISIK"
        elif "KELENGKAPAN" in c: rename_map[c] = "KELENGKAPAN"
        elif "LABEL" in c: rename_map[c] = "LABEL_KONDISI"
    
    df_raw.rename(columns=rename_map, inplace=True)
    df = df_raw.copy()
    
    # Preprocessing
    for col in FITUR_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df_train = df[df[LABEL_COL].notna() & (df[LABEL_COL] != "TIDAK DIKETAHUI")].copy()
    df_train[LABEL_COL] = df_train[LABEL_COL].astype(str).str.strip().str.upper()
    
    le = LabelEncoder()
    df_train["LABEL_ENC"] = le.fit_transform(df_train[LABEL_COL])
    
    X = df_train[FITUR_COLS]
    y = df_train["LABEL_ENC"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )
    
    gnb = GaussianNB()
    bnb = BernoulliNB()
    gnb.fit(X_train, y_train)
    bnb.fit(X_train, y_train)
    
    y_pred_gnb = gnb.predict(X_test)
    y_pred_bnb = bnb.predict(X_test)
    
    return (df, df_train, le, gnb, bnb, X_train, X_test, y_train, y_test, y_pred_gnb, y_pred_bnb)

@st.cache_data
def load_notebook():
    """Load notebook Jupyter"""
    try:
        if os.path.exists("naive-bayes.ipynb"):
            with open("naive-bayes.ipynb", "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    except:
        return None

# ─────────────────────────────────────────────────────────────
# INITIALIZE APP - AUTO LOAD
# ─────────────────────────────────────────────────────────────
st.markdown("# 📦 **Inventaris Naive Bayes**")
st.markdown("**SMK Muhammadiyah 12 - Tahun 2025**")

# Load data otomatis
df_raw = load_data()
model_data = preprocess_and_train(df_raw)
notebook = load_notebook()

(df, df_train, le, gnb, bnb, X_train, X_test, y_train, y_test, y_pred_gnb, y_pred_bnb) = model_data
label_counts = df_train[LABEL_COL].value_counts()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 **Inventaris NB**")
    st.markdown("**SMK Muhammadiyah 12**")
    st.markdown("**Tahun 2025**")
    st.markdown("---")
    st.success("✅ Data Loaded!")
    st.metric("Total Barang", len(df))
    st.metric("Data Training", len(df_train))
    st.markdown("---")
    
    menu = st.radio("📂 Menu:", [
        "🏠 Dashboard",
        "📋 Data Inventaris", 
        "🤖 Model Evaluasi",
        "📊 Confusion Matrix",
        "🔮 Prediksi Baru",
        "📓 Notebook Code"
    ])

# ─────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────

if menu == "🏠 Dashboard":
    st.markdown("## 🏠 **Dashboard Utama**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(metric_card(len(df), "Total Barang", "blue"), unsafe_allow_html=True)
    with col2: st.markdown(metric_card(label_counts.get("LAYAK", 0), "Layak", "green"), unsafe_allow_html=True)
    with col3: st.markdown(metric_card(label_counts.get("KURANG LAYAK", 0), "Kurang Layak", "yellow"), unsafe_allow_html=True)
    with col4: st.markdown(metric_card(label_counts.get("TIDAK LAYAK", 0), "Tidak Layak", "red"), unsafe_allow_html=True)
    
    col_left, col_right = st.columns([2,1])
    
    with col_left:
        section("Distribusi Kondisi Barang")
        fig, ax = plt.subplots(figsize=(8,5))
        labels = label_counts.index
        values = label_counts.values
        colors = [COLOR_MAP.get(l, "#999") for l in labels]
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=2)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.1, 
                   str(v), ha='center', va='bottom', fontweight='bold')
        ax.set_title("Distribusi Kondisi Inventaris", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col_right:
        section("Akurasi Model")
        acc_gnb = accuracy_score(y_test, y_pred_gnb)*100
        acc_bnb = accuracy_score(y_test, y_pred_bnb)*100
        st.markdown(metric_card(f"{acc_gnb:.1f}%", "Gaussian NB", "blue"), unsafe_allow_html=True)
        st.markdown(metric_card(f"{acc_bnb:.1f}%", "Bernoulli NB", "blue"), unsafe_allow_html=True)

elif menu == "📋 Data Inventaris":
    st.markdown("## 📋 **Data Inventaris Lengkap**")
    
    col1, col2, col3 = st.columns(3)
    with col1: filter_label = st.selectbox("Kondisi:", ["Semua"] + sorted(df_train[LABEL_COL].unique()))
    with col2: filter_kat = st.selectbox("Kategori:", ["Semua"] + sorted(df_train.get("KATEGORI_BARANG", pd.Series()).unique()))
    with col3: search = st.text_input("Cari barang:")
    
    df_view = df_train.copy()
    if filter_label != "Semua": df_view = df_view[df_view[LABEL_COL] == filter_label]
    if filter_kat != "Semua" and "KATEGORI_BARANG" in df_view: df_view = df_view[df_view["KATEGORI_BARANG"] == filter_kat]
    if search: df_view = df_view[df_view["NAMA_BARANG"].str.contains(search, case=False, na=False)]
    
    st.info(f"Menampilkan **{len(df_view)}** dari **{len(df_train)}** data")
    
    show_cols = [c for c in INFO_COLS + FITUR_COLS + [LABEL_COL] if c in df_view.columns]
    st.dataframe(df_view[show_cols], use_container_width=True, height=600)

elif menu == "🤖 Model Evaluasi":
    st.markdown("## 🤖 **Evaluasi Model Naive Bayes**")
    
    tab1, tab2 = st.tabs(["Gaussian NB", "Bernoulli NB"])
    
    for tab, model, name, y_pred in zip([tab1, tab2], [gnb, bnb], ["Gaussian NB", "Bernoulli NB"], [y_pred_gnb, y_pred_bnb]):
        with tab:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.markdown(metric_card(f"{acc*100:.1f}%", "Accuracy", "blue"), unsafe_allow_html=True)
            with col2: st.markdown(metric_card(f"{prec:.3f}", "Precision", "green"), unsafe_allow_html=True)
            with col3: st.markdown(metric_card(f"{rec:.3f}", "Recall", "yellow"), unsafe_allow_html=True)
            with col4: st.markdown(metric_card(f"{f1:.3f}", "F1-Score", "blue"), unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).T.round(3))

elif menu == "📊 Confusion Matrix":
    st.markdown("## 📊 **Confusion Matrix**")
    model_choice = st.selectbox("Model:", ["Gaussian NB", "Bernoulli NB"])
    y_pred = y_pred_gnb if "Gaussian" in model_choice else y_pred_bnb
    
    fig, ax = plt.subplots(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_title(f"Confusion Matrix - {model_choice}", fontsize=14, fontweight='bold')
    st.pyplot(fig)

elif menu == "🔮 Prediksi Baru":
    st.markdown("## 🔮 **Prediksi Kondisi Baru**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📝 Input Data")
        nama = st.text_input("Nama Barang")
        model_sel = st.selectbox("Model", ["Gaussian NB", "Bernoulli NB"])
        tahun = st.slider("Tahun Pengadaan", 2015, 2025, 2023)
        frek = st.slider("Frekuensi (1=Jarang,5=Sering)", 1, 5, 3)
        umur = st.slider("Umur (tahun)", 0, 10, 2)
        fisik = st.slider("Kondisi Fisik (1=Rusak,5=Baik)", 1, 5, 4)
        lengkap = st.slider("Kelengkapan (1=Tidak,5=Lengkap)", 1, 5, 4)
        
        if st.button("🔮 **Prediksi**", use_container_width=True):
            input_data = np.array([[tahun, frek, umur, fisik, lengkap]])
            model = gnb if "Gaussian" in model_sel else bnb
            pred = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            
            with col2:
                st.subheader("📊 Hasil")
                pred_label = le.inverse_transform([pred])[0]
                st.markdown(f"**Prediksi:** {badge(pred_label)}", unsafe_allow_html=True)
                
                st.subheader("Probabilitas")
                for i, (cls, p) in enumerate(zip(le.classes_, proba)):
                    col_color = "success" if i == pred else "secondary"
                    st.metric(cls, f"{p*100:.1f}%")
        else:
            with col2:
                    st.info("👈 Isi form di kiri untuk prediksi")

elif menu == "📓 Notebook Code":
    st.markdown("## 📓 **naive-bayes.ipynb**")
    if notebook:
        for cell in notebook["cells"]:
            if cell["cell_type"] == "markdown":
                st.markdown("".join(cell["source"]))
            elif cell["cell_type"] == "code":
                st.code("".join(cell["source"]), language="python")
    else:
        st.warning("File naive-bayes.ipynb tidak ditemukan")