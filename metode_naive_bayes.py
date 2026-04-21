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

st.markdown("""
<style>
    .stApp { background-color: #F0F4F8; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1F4E79 0%, #2E75B6 100%); }
    section[data-testid="stSidebar"] * { color: white !important; }
    .metric-card { background: white; border-radius: 12px; padding: 20px 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 5px solid #2E75B6; margin-bottom: 12px; }
    .metric-card.green  { border-left-color: #70AD47; }
    .metric-card.yellow { border-left-color: #FFC000; }
    .metric-card.red    { border-left-color: #FF0000; }
    .metric-card.blue   { border-left-color: #2E75B6; }
    .metric-val  { font-size: 2rem; font-weight: 700; color: #1F4E79; }
    .metric-label{ font-size: 0.85rem; color: #666; margin-top: 4px; }
    .section-header { background: linear-gradient(90deg, #1F4E79, #2E75B6); color: white; padding: 10px 18px; border-radius: 8px; font-size: 1.05rem; font-weight: 600; margin: 20px 0 14px 0; }
    .badge-layak    { background:#70AD47; color:white; padding:6px 16px; border-radius:20px; font-weight:700; }
    .badge-kurang   { background:#FFC000; color:white; padding:6px 16px; border-radius:20px; font-weight:700; }
    .badge-tidak    { background:#FF0000; color:white; padding:6px 16px; border-radius:20px; font-weight:700; }
    hr { border: 1px solid #D0D7DE; margin: 18px 0; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS - FIXED LABEL NAME
# ─────────────────────────────────────────────────────────────
LABEL_COL = "Label Kondisi"  # ✅ DIUBAH SESUAI REQUEST
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
    """✅ SAFE LOAD CSV dengan error handling"""
    csv_path = "Data Inventaris SMK Muhammadiyah 12 - Tahun 2025.csv"
    try:
        if not os.path.exists(csv_path):
            st.error(f"❌ **File CSV TIDAK DITEMUKAN**: {csv_path}")
            st.stop()
        
        df_raw = pd.read_csv(csv_path)
        st.sidebar.success(f"✅ Loaded: **{len(df_raw):,d}** rows")
        return df_raw
    except Exception as e:
        st.error(f"❌ **Error loading CSV**: {str(e)}")
        st.stop()

@st.cache_data
def preprocess_and_train(df_raw):
    """✅ ROBUST PREPROCESSING - Handle missing columns"""
    print("DEBUG: Columns awal:", df_raw.columns.tolist())
    
    # Normalize column names
    df_raw.columns = [str(c).strip().upper().replace(" ", "_") for c in df_raw.columns]
    print("DEBUG: Columns setelah normalize:", df_raw.columns.tolist())
    
    # Flexible column mapping ✅
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
        elif any(x in c for x in ["LABEL", "LABEL_KONDISI", "LABEL KONDISI"]): 
            rename_map[c] = "Label Kondisi"  # ✅ EXACT MATCH
    
    df_raw.rename(columns=rename_map, inplace=True)
    print("DEBUG: Columns setelah rename:", df_raw.columns.tolist())
    
    # Check required columns
    if "Label Kondisi" not in df_raw.columns:
        st.error("❌ **Kolom 'Label Kondisi' TIDAK DITEMUKAN**")
        st.write("Kolom tersedia:", df_raw.columns.tolist())
        st.stop()
    
    df = df_raw.copy()
    
    # Safe numeric conversion
    available_features = [col for col in FITUR_COLS if col in df.columns]
    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Filter training data ✅ SAFE FILTER
    df_train = df[df["Label Kondisi"].notna()].copy()
    if len(df_train) == 0:
        st.error("❌ **Tidak ada data dengan Label Kondisi yang valid**")
        st.stop()
    
    df_train["Label Kondisi"] = df_train["Label Kondisi"].astype(str).str.strip().str.upper()
    
    # Label encoding
    le = LabelEncoder()
    df_train["LABEL_ENC"] = le.fit_transform(df_train["Label Kondisi"])
    
    X = df_train[available_features]
    y = df_train["LABEL_ENC"]
    
    if len(y.unique()) < 2:
        st.warning("⚠️ Hanya 1 kelas ditemukan, menggunakan full data untuk training")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train models
    gnb = GaussianNB()
    bnb = BernoulliNB()
    gnb.fit(X_train, y_train)
    bnb.fit(X_train, y_train)
    
    y_pred_gnb = gnb.predict(X_test)
    y_pred_bnb = bnb.predict(X_test)
    
    print(f"✅ Training selesai. Features: {available_features}, Classes: {le.classes_}")
    return (df, df_train, le, gnb, bnb, X_train, X_test, y_train, y_test, y_pred_gnb, y_pred_bnb, available_features)

@st.cache_data
def load_notebook():
    try:
        if os.path.exists("naive-bayes.ipynb"):
            with open("naive-bayes.ipynb", "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    except:
        return None

# ─────────────────────────────────────────────────────────────
# MAIN APP - AUTO LOAD ✅
# ─────────────────────────────────────────────────────────────
try:
    st.title("📦 **Sistem Inventaris Naive Bayes**")
    st.markdown("**SMK Muhammadiyah 12 - Tahun 2025**")
    
    # Load everything
    df_raw = load_data()
    model_data = preprocess_and_train(df_raw)
    notebook = load_notebook()
    
    (df, df_train, le, gnb, bnb, X_train, X_test, y_train, y_test, y_pred_gnb, y_pred_bnb, feature_cols) = model_data
    label_counts = df_train["Label Kondisi"].value_counts()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📦 **Inventaris NB**")
        st.markdown("**SMK Muhammadiyah 12**")
        col1, col2 = st.columns(2)
        with col1: st.metric("Total", len(df))
        with col2: st.metric("Training", len(df_train))
        st.markdown("---")
        
        menu = st.radio("📂 Pilih:", [
            "🏠 Dashboard",
            "📋 Data", 
            "🤖 Model", 
            "📊 Matrix",
            "🔮 Prediksi",
            "📓 Notebook"
        ])
    
    # Pages
    if menu == "🏠 Dashboard":
        st.markdown("### 🏠 **Dashboard**")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(metric_card(len(df), "Total", "blue"), unsafe_allow_html=True)
        with c2: st.markdown(metric_card(label_counts.get("LAYAK", 0), "Layak", "green"), unsafe_allow_html=True)
        with c3: st.markdown(metric_card(label_counts.get("KURANG LAYAK", 0), "Kurang", "yellow"), unsafe_allow_html=True)
        with c4: st.markdown(metric_card(label_counts.get("TIDAK LAYAK", 0), "Tidak Layak", "red"), unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            section("Distribusi")
            fig, ax = plt.subplots(figsize=(8,5))
            ax.bar(label_counts.index, label_counts.values, color=[COLOR_MAP.get(l, "#ccc") for l in label_counts.index])
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            section("Akurasi")
            st.markdown(metric_card(f"{accuracy_score(y_test, y_pred_gnb)*100:.1f}%", "Gaussian NB", "blue"), unsafe_allow_html=True)
            st.markdown(metric_card(f"{accuracy_score(y_test, y_pred_bnb)*100:.1f}%", "Bernoulli NB", "blue"), unsafe_allow_html=True)
    
    elif menu == "📋 Data":
        st.markdown("### 📋 **Data Inventaris**")
        filter_label = st.selectbox("Kondisi:", ["Semua"] + sorted(df_train["Label Kondisi"].unique()))
        search = st.text_input("🔍 Cari:")
        
        df_view = df_train.copy()
        if filter_label != "Semua": 
            df_view = df_view[df_view["Label Kondisi"] == filter_label]
        if search: 
            df_view = df_view[df_view["NAMA_BARANG"].str.contains(search, case=False, na=False)]
        
        st.dataframe(df_view, use_container_width=True)
    
    elif menu == "🤖 Model":
        st.markdown("### 🤖 **Evaluasi Model**")
        acc_gnb = accuracy_score(y_test, y_pred_gnb)
        st.success(f"**Gaussian NB Accuracy: {acc_gnb*100:.2f}%**")
        st.dataframe(pd.DataFrame({
            'Model': ['Gaussian NB', 'Bernoulli NB'],
            'Accuracy': [acc_gnb*100, accuracy_score(y_test, y_pred_bnb)*100]
        }))
    
    elif menu == "📊 Matrix":
        st.markdown("### 📊 **Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred_gnb)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
    
    elif menu == "🔮 Prediksi":
        st.markdown("### 🔮 **Prediksi Baru**")
        col1, col2 = st.columns(2)
        with col1:
            nama = st.text_input("Nama Barang:")
            tahun = st.slider("Tahun", 2015, 2025, 2023)
            frek = st.slider("Frekuensi", 1, 5, 3)
            umur = st.slider("Umur", 0, 10, 2)
            fisik = st.slider("Fisik", 1, 5, 4)
            lengkap = st.slider("Kelengkap", 1, 5, 4)
            
            if st.button("🔮 Prediksi"):
                input_data = np.array([[tahun, frek, umur, fisik, lengkap]])
                pred = gnb.predict(input_data)[0]
                pred_label = le.inverse_transform([pred])[0]
                
                with col2:
                    st.markdown(f"**Hasil: {badge(pred_label)}**", unsafe_allow_html=True)
    
    elif menu == "📓 Notebook":
        st.markdown("### 📓 **naive-bayes.ipynb**")
        if notebook:
            for cell in notebook["cells"][:10]:  # Limit to first 10 cells
                if cell["cell_type"] == "markdown":
                    st.markdown("".join(cell["source"]))
                elif cell["cell_type"] == "code":
                    st.code("".join(cell["source"]))
        else:
            st.info("📓 File notebook tidak ditemukan")

except Exception as e:
    st.error(f"❌ **Error**: {str(e)}")
    st.stop()