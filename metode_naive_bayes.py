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
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"]  { color: #000000 !important; }
.stApp { background-color: #F0F4F8; color: #000000 !important; }
h1, h2, h3, h4, h5, h6,
p, span, div, label, strong, small { color: #000000 !important; opacity: 1 !important; }
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 { color: #000000 !important; }
.stMarkdown { color: #000000 !important; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1F4E79 0%, #2E75B6 100%);
}
section[data-testid="stSidebar"] * { color: white !important; }
* { opacity: 1 !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
LABEL_COL    = "Label Kondisi"
FITUR_COLS   = ["TAHUN_PENGADAAN", "FREKUENSI_PEMAKAIAN", "UMUR_BARANG", "KONDISI_FISIK", "KELENGKAPAN"]
INFO_COLS    = ["NAMA_BARANG", "MERK", "KODE_BARANG", "KATEGORI_BARANG"]
COLOR_MAP    = {"LAYAK": "#70AD47", "KURANG LAYAK": "#FFC000", "TIDAK LAYAK": "#FF0000"}
KNOWN_LABELS = {"LAYAK", "KURANG LAYAK", "TIDAK LAYAK", "KURANG_LAYAK", "TIDAK_LAYAK"}

def badge(label):
    cls = {"LAYAK": "badge-layak", "KURANG LAYAK": "badge-kurang", "TIDAK LAYAK": "badge-tidak"}.get(label, "")
    return f'<span class="{cls}">{label}</span>'

def metric_card(val, label, color="blue"):
    return f'''
    <div class="metric-card {color}">
        <div class="metric-val">{val}</div>
        <div class="metric-label">{label}</div>
    </div>'''

def section(title):
    st.markdown(f'<div class="section-header">📌 {title}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = "Data Inventaris SMK Muhammadiyah 12 - Tahun 2025.csv"
    try:
        if not os.path.exists(csv_path):
            st.error(f"❌ **File CSV TIDAK DITEMUKAN**: `{csv_path}`")
            st.stop()
        try:
            df_raw = pd.read_csv(csv_path)
        except Exception:
            df_raw = pd.read_csv(csv_path, header=1)

        st.sidebar.success(f"✅ Loaded: **{len(df_raw):,d}** rows")
        return df_raw
    except Exception as e:
        st.error(f"❌ **Error loading CSV**: {str(e)}")
        st.stop()

# ─────────────────────────────────────────────────────────────
# PREPROCESS & TRAIN  (semua fix diterapkan di sini)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def preprocess_and_train(df_raw):
    df_raw = df_raw.copy()

    # ── 1. Normalisasi nama kolom ─────────────────────────────
    df_raw.columns = [str(c).strip().upper().replace(" ", "_") for c in df_raw.columns]

    # ── 2. Mapping kolom berdasarkan nama ─────────────────────
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
        elif "FISIK"       in cu:                    rename_map[c] = "KONDISI_FISIK"
        elif "KELENGKAPAN" in cu:                    rename_map[c] = "KELENGKAPAN"
        elif any(x in cu for x in ["LABEL", "STATUS", "KETERANGAN", "KLASIFIKASI"]):
            rename_map[c] = "Label Kondisi"

    df_raw.rename(columns=rename_map, inplace=True)

    # ── 3. FIX #1 — Auto-detect kolom label dari ISI data ────
    if "Label Kondisi" not in df_raw.columns:
        assigned_cols = set(rename_map.values()) | set(FITUR_COLS) | set(INFO_COLS)
        for col in df_raw.columns:
            if col in assigned_cols:
                continue
            unique_vals = set(
                df_raw[col].dropna().astype(str)
                .str.strip().str.upper().unique()
            )
            if unique_vals & KNOWN_LABELS:
                df_raw.rename(columns={col: "Label Kondisi"}, inplace=True)
                st.sidebar.info(f"🔍 Kolom label terdeteksi: **{col}**")
                break

    # ── 4. Gagal total → debug message ───────────────────────
    if "Label Kondisi" not in df_raw.columns:
        st.error("❌ Kolom **'Label Kondisi'** tidak ditemukan!")
        st.warning("📋 Kolom yang tersedia di CSV:")
        st.code("\n".join(df_raw.columns.tolist()))
        st.info("💡 Pastikan salah satu kolom berisi nilai: **LAYAK / KURANG LAYAK / TIDAK LAYAK**")
        st.stop()

    df = df_raw.copy()

    # ── 5. Konversi fitur numerik ─────────────────────────────
    available_features = [col for col in FITUR_COLS if col in df.columns]
    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ── 6. Bersihkan label ────────────────────────────────────
    df_train = df[df["Label Kondisi"].notna()].copy()
    df_train["Label Kondisi"] = (
        df_train["Label Kondisi"].astype(str).str.strip().str.upper()
    )

    # ── 7. Tampilkan distribusi label di sidebar ──────────────
    label_dist = df_train["Label Kondisi"].value_counts()
    st.sidebar.markdown("### 📊 Distribusi Label")
    for lbl, cnt in label_dist.items():
        icon = "✅" if cnt >= 5 else ("⚠️" if cnt >= 2 else "❌")
        st.sidebar.markdown(f"{icon} **{lbl}**: {cnt} data")

    # ── 8. FIX #2 — Hapus kelas yang hanya punya 1 anggota ───
    valid_labels = label_dist[label_dist >= 2].index
    removed_labels = label_dist[label_dist < 2].index.tolist()

    if removed_labels:
        st.sidebar.error(f"🗑️ Kelas dihapus (< 2 data): **{removed_labels}**")
        st.warning(
            f"⚠️ Kelas **{removed_labels}** dihapus dari training "
            "karena jumlah data terlalu sedikit (< 2). "
            "Tambahkan lebih banyak data untuk kelas tersebut."
        )
        df_train = df_train[df_train["Label Kondisi"].isin(valid_labels)].copy()

    # ── 9. Encode label ───────────────────────────────────────
    le = LabelEncoder()
    df_train["LABEL_ENC"] = le.fit_transform(df_train["Label Kondisi"])

    X = df_train[available_features]
    y = df_train["LABEL_ENC"]

    # ── 10. FIX #2 — Stratify hanya jika semua kelas >= 2 ────
    class_counts  = y.value_counts()
    use_stratify  = y if class_counts.min() >= 2 else None

    if use_stratify is None:
        st.sidebar.warning("⚠️ Stratify dinonaktifkan\n(ada kelas dengan data < 2)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=use_stratify
    )

    # ── 11. Training model ────────────────────────────────────
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

    df_raw     = load_data()
    model_data = preprocess_and_train(df_raw)

    (df, df_train, le, gnb, bnb,
     X_train, X_test, y_train, y_test,
     y_pred_gnb, y_pred_bnb, feature_cols) = model_data

    label_counts = df_train["Label Kondisi"].value_counts()

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📦 **Inventaris NB**")
        st.markdown("**SMK Muhammadiyah 12**")
        st.metric("Total Barang",   len(df))
        st.metric("Data Training",  len(df_train))
        st.markdown("---")
        menu = st.radio("📂 Menu:", [
            "🏠 Dashboard",
            "📋 Data Inventaris",
            "🤖 Model Evaluasi",
            "📊 Confusion Matrix",
            "🔮 Prediksi Baru"
        ])

    # ══════════════════════════════════════════════════════════
    # PAGE: Dashboard
    # ══════════════════════════════════════════════════════════
    if menu == "🏠 Dashboard":
        st.markdown("### 🏠 **Dashboard Utama**")

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(metric_card(f"{len(df):,d}", "Total Barang", "blue"),        unsafe_allow_html=True)
        with col2: st.markdown(metric_card(label_counts.get("LAYAK", 0), "Layak ✅", "green"),          unsafe_allow_html=True)
        with col3: st.markdown(metric_card(label_counts.get("KURANG LAYAK", 0), "Kurang ⚠️", "yellow"), unsafe_allow_html=True)
        with col4: st.markdown(metric_card(label_counts.get("TIDAK LAYAK", 0), "Tidak Layak ❌", "red"),unsafe_allow_html=True)

        col_left, col_right = st.columns([2, 1])
        with col_left:
            section("📊 Distribusi Kondisi")
            fig, ax = plt.subplots(figsize=(9, 5))
            colors = [COLOR_MAP.get(l, "#ccc") for l in label_counts.index]
            ax.bar(label_counts.index, label_counts.values, color=colors, edgecolor='white', linewidth=2)
            ax.set_title("Distribusi Barang Inventaris", fontweight='bold', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

        with col_right:
            section("🎯 Akurasi Model")
            acc_gnb = accuracy_score(y_test, y_pred_gnb) * 100
            acc_bnb = accuracy_score(y_test, y_pred_bnb) * 100
            st.markdown(metric_card(f"{acc_gnb:.1f}%", "Gaussian NB",  "blue"), unsafe_allow_html=True)
            st.markdown(metric_card(f"{acc_bnb:.1f}%", "Bernoulli NB", "blue"), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # PAGE: Data Inventaris
    # ══════════════════════════════════════════════════════════
    elif menu == "📋 Data Inventaris":
        st.markdown("### 📋 **Data Inventaris Lengkap**")

        col1, col2 = st.columns(2)
        with col1:
            filter_label = st.selectbox(
                "Filter Kondisi:",
                ["Semua"] + sorted(df_train["Label Kondisi"].unique().tolist())
            )
        with col2:
            search = st.text_input("🔍 Cari nama barang:", placeholder="Ketik nama barang...")

        df_view = df_train.copy()
        if filter_label != "Semua":
            df_view = df_view[df_view["Label Kondisi"] == filter_label]
        if search and "NAMA_BARANG" in df_view.columns:
            df_view = df_view[df_view["NAMA_BARANG"].str.contains(search, case=False, na=False)]

        st.info(f"**Menampilkan {len(df_view):,d} dari {len(df_train):,d} data**")
        st.dataframe(df_view, use_container_width=True, height=600)

    # ══════════════════════════════════════════════════════════
    # PAGE: Model Evaluasi
    # ══════════════════════════════════════════════════════════
    elif menu == "🤖 Model Evaluasi":
        st.markdown("### 🤖 **Evaluasi Model Naive Bayes**")

        col1, col2 = st.columns(2)
        with col1:
            section("Gaussian Naive Bayes")
            acc_g  = accuracy_score(y_test, y_pred_gnb)
            prec_g = precision_score(y_test, y_pred_gnb, average='weighted', zero_division=0)
            rec_g  = recall_score(y_test, y_pred_gnb, average='weighted', zero_division=0)
            f1_g   = f1_score(y_test, y_pred_gnb, average='weighted', zero_division=0)
            st.metric("Accuracy",  f"{acc_g*100:.2f}%")
            st.metric("Precision", f"{prec_g*100:.2f}%")
            st.metric("Recall",    f"{rec_g*100:.2f}%")
            st.metric("F1-Score",  f"{f1_g*100:.2f}%")

        with col2:
            section("Bernoulli Naive Bayes")
            acc_b  = accuracy_score(y_test, y_pred_bnb)
            prec_b = precision_score(y_test, y_pred_bnb, average='weighted', zero_division=0)
            rec_b  = recall_score(y_test, y_pred_bnb, average='weighted', zero_division=0)
            f1_b   = f1_score(y_test, y_pred_bnb, average='weighted', zero_division=0)
            st.metric("Accuracy",  f"{acc_b*100:.2f}%")
            st.metric("Precision", f"{prec_b*100:.2f}%")
            st.metric("Recall",    f"{rec_b*100:.2f}%")
            st.metric("F1-Score",  f"{f1_b*100:.2f}%")

        st.markdown("---")
        st.markdown("#### 📄 Classification Report")
        model_rep = st.selectbox("Pilih Model Report:", ["Gaussian NB", "Bernoulli NB"])
        y_rep = y_pred_gnb if model_rep == "Gaussian NB" else y_pred_bnb
        report = classification_report(y_test, y_rep, target_names=le.classes_, zero_division=0)
        st.code(report)

    # ══════════════════════════════════════════════════════════
    # PAGE: Confusion Matrix
    # ══════════════════════════════════════════════════════════
    elif menu == "📊 Confusion Matrix":
        st.markdown("### 📊 **Confusion Matrix**")
        model_choice = st.selectbox("Pilih Model:", ["Gaussian NB", "Bernoulli NB"])
        y_pred = y_pred_gnb if model_choice == "Gaussian NB" else y_pred_bnb

        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_title(f"Confusion Matrix - {model_choice}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        st.pyplot(fig)

    # ══════════════════════════════════════════════════════════
    # PAGE: Prediksi Baru
    # ══════════════════════════════════════════════════════════
    elif menu == "🔮 Prediksi Baru":
        st.markdown("### 🔮 **Prediksi Kondisi Barang Baru**")

        col_form, col_result = st.columns([1, 1])

        with col_form:
            section("📝 Input Data")
            nama_barang   = st.text_input("Nama Barang:", placeholder="Contoh: Laptop Dell")
            model_pilihan = st.selectbox("Model:", ["Gaussian NB", "Bernoulli NB"])

            st.markdown("**Fitur Numerik:**")
            tahun        = st.slider("📅 Tahun Pengadaan", 2015, 2025, 2023)
            frekuensi    = st.slider("⏰ Frekuensi Pemakaian (1=Jarang, 5=Sering)", 1, 5, 3)
            umur         = st.slider("📏 Umur Barang (tahun)", 0, 10, 2)
            kondisi_fisik= st.slider("🔧 Kondisi Fisik (1=Rusak, 5=Baik)", 1, 5, 4)
            kelengkapan  = st.slider("📦 Kelengkapan (1=Tidak lengkap, 5=Lengkap)", 1, 5, 4)

            predict_btn = st.button("🔮 **Lakukan Prediksi**", use_container_width=True, type="primary")

        # Tampilkan hasil di luar button agar kolom kanan selalu terrender
        with col_result:
            if predict_btn:
                # Sesuaikan urutan input dengan feature_cols yang tersedia
                input_vals = {
                    "TAHUN_PENGADAAN":    tahun,
                    "FREKUENSI_PEMAKAIAN": frekuensi,
                    "UMUR_BARANG":        umur,
                    "KONDISI_FISIK":      kondisi_fisik,
                    "KELENGKAPAN":        kelengkapan,
                }
                input_data = np.array([[input_vals[f] for f in feature_cols]])

                model      = gnb if model_pilihan == "Gaussian NB" else bnb
                prediction = model.predict(input_data)[0]
                pred_label = le.inverse_transform([prediction])[0]
                proba      = model.predict_proba(input_data)[0]

                section("📊 Hasil Prediksi")
                st.markdown(f"**Barang:** {nama_barang or 'Tidak disebutkan'}")
                st.markdown(f"**Prediksi:** {badge(pred_label)}", unsafe_allow_html=True)
                st.success(f"Model: **{model_pilihan}**")
                st.info(
                    f"Input → Tahun: **{tahun}** | Frekuensi: **{frekuensi}** | "
                    f"Umur: **{umur}** | Fisik: **{kondisi_fisik}** | Lengkap: **{kelengkapan}**"
                )

                # Probabilitas per kelas
                st.markdown("**🎲 Probabilitas per Kelas:**")
                proba_df = pd.DataFrame({
                    "Kelas": le.classes_,
                    "Probabilitas (%)": [f"{p*100:.2f}%" for p in proba]
                })
                st.dataframe(proba_df, use_container_width=True, hide_index=True)
            else:
                st.info("👈 Isi form di kiri lalu klik **Lakukan Prediksi**")

except Exception as e:
    st.error(f"❌ **Terjadi Error**: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
    st.info("Pastikan file CSV ada di folder yang sama dengan metode_naive_bayes.py")