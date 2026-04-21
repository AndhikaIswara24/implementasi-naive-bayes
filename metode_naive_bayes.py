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

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inventaris Naive Bayes",
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
def load_and_train(uploaded):
    df_raw = pd.read_csv(uploaded)

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
        elif "FREK" in c:      rename[c] = "FREKUENSI_PEMAKAIAN"
        elif "UMUR" in c:      rename[c] = "UMUR_BARANG"
        elif "FISIK" in c or "KONDISI_F" in c: rename[c] = "KONDISI_FISIK"
        elif "LENGKAP" in c:   rename[c] = "KELENGKAPAN"
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

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 Inventaris NB")
    st.markdown("---")
    uploaded = st.file_uploader("Upload file CSV inventaris", type=["csv"])
    st.markdown("---")
    st.markdown("### 🔍 Navigasi")
    menu = st.radio("Pilih halaman:", [
        "🏠 Dashboard",
        "📋 Data Inventaris",
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
# MAIN
# ─────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px;'>
        <h1 style='color:#1F4E79;'>📦 Sistem Klasifikasi Inventaris</h1>
        <h3 style='color:#2E75B6;'>Metode Naive Bayes</h3>
        <p style='color:#666; font-size:1.1rem;'>
            Implementasi algoritma Naive Bayes untuk memprediksi<br>
            kelayakan barang inventaris berdasarkan kondisi fisik dan pemakaian.
        </p>
        <br>
        <div style='background:white; border-radius:12px; padding:30px; 
                    box-shadow:0 4px 16px rgba(0,0,0,0.1); max-width:500px; margin:auto;'>
            <h4 style='color:#1F4E79;'>⬅️ Upload CSV di sidebar untuk memulai</h4>
            <p style='color:#888; font-size:0.9rem;'>
                Pastikan CSV memiliki kolom:<br>
                <code>NAMA_BARANG, MERK, KODE_BARANG, KATEGORI_BARANG,<br>
                TAHUN_PENGADAAN, FREKUENSI_PEMAKAIAN, UMUR_BARANG,<br>
                KONDISI_FISIK, KELENGKAPAN, LABEL_KONDISI</code>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load & Train ────────────────────────────────────────────
try:
    (df, df_train, le, gnb, bnb,
     X_train, X_test, y_train, y_test,
     y_pred_gnb, y_pred_bnb) = load_and_train(uploaded)
except Exception as e:
    st.error(f"❌ Gagal memproses CSV: {e}")
    st.stop()

label_counts = df_train[LABEL_COL].value_counts()
best_model   = gnb  # default

# ═══════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════
if menu == "🏠 Dashboard":
    st.markdown("<h2 style='color:#1F4E79;'>🏠 Dashboard Inventaris</h2>", unsafe_allow_html=True)
    st.markdown("Ringkasan kondisi barang inventaris menggunakan klasifikasi **Naive Bayes**.")
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

    section("Distribusi Barang per Kategori")
    if "KATEGORI_BARANG" in df_train.columns:
        cat_data = df_train.groupby(["KATEGORI_BARANG", LABEL_COL]).size().unstack(fill_value=0)
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        cat_data.plot(kind='bar', ax=ax3, color=[COLOR_MAP.get(c,"#999") for c in cat_data.columns],
                      edgecolor='white', linewidth=1)
        ax3.set_title("Jumlah Barang per Kategori & Kondisi", fontweight='bold')
        ax3.set_xlabel(""); ax3.set_ylabel("Jumlah")
        ax3.legend(title="Kondisi", bbox_to_anchor=(1.01,1))
        ax3.tick_params(axis='x', rotation=30)
        ax3.set_facecolor("#F8F9FA"); fig3.patch.set_facecolor("white")
        ax3.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    section("Akurasi Model (ringkasan)")
    m1, m2 = st.columns(2)
    acc_g = accuracy_score(y_test, y_pred_gnb) * 100
    acc_b = accuracy_score(y_test, y_pred_bnb) * 100
    with m1:
        st.markdown(metric_card(f"{acc_g:.2f}%", "Akurasi Gaussian Naive Bayes", "blue"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card(f"{acc_b:.2f}%", "Akurasi Bernoulli Naive Bayes", "blue"), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE: DATA INVENTARIS
# ═══════════════════════════════════════════════════════════
elif menu == "📋 Data Inventaris":
    st.markdown("<h2 style='color:#1F4E79;'>📋 Data Inventaris</h2>", unsafe_allow_html=True)
    st.markdown("---")

    # Filter
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_label = st.selectbox("Filter Label:", ["Semua"] + df_train[LABEL_COL].unique().tolist())
    with col_f2:
        filter_kat   = st.selectbox("Filter Kategori:", ["Semua"] + (
            df_train["KATEGORI_BARANG"].unique().tolist() if "KATEGORI_BARANG" in df_train.columns else []))
    with col_f3:
        search = st.text_input("🔍 Cari nama barang:", "")

    df_view = df_train.copy()
    if filter_label != "Semua":
        df_view = df_view[df_view[LABEL_COL] == filter_label]
    if filter_kat != "Semua" and "KATEGORI_BARANG" in df_view.columns:
        df_view = df_view[df_view["KATEGORI_BARANG"] == filter_kat]
    if search:
        df_view = df_view[df_view["NAMA_BARANG"].str.contains(search, case=False, na=False)]

    st.markdown(f"**Menampilkan {len(df_view)} dari {len(df_train)} data latih**")

    show_cols = [c for c in INFO_COLS + FITUR_COLS + [LABEL_COL] if c in df_view.columns]

    def color_label(val):
        colors = {"LAYAK":"background-color:#C6EFCE;color:#276221",
                  "KURANG LAYAK":"background-color:#FFEB9C;color:#7F6000",
                  "TIDAK LAYAK":"background-color:#FFC7CE;color:#9C0006"}
        return colors.get(val, "")

    styled = df_view[show_cols].reset_index(drop=True).style.applymap(
        color_label, subset=[LABEL_COL])
    st.dataframe(styled, use_container_width=True, height=480)

    section("Statistik Deskriptif Fitur Numerik")
    st.dataframe(df_train[FITUR_COLS].describe().round(2), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE: MODEL & EVALUASI
# ═══════════════════════════════════════════════════════════
elif menu == "🤖 Model & Evaluasi":
    st.markdown("<h2 style='color:#1F4E79;'>🤖 Evaluasi Model Naive Bayes</h2>", unsafe_allow_html=True)
    st.markdown("---")

    section("Pembagian Data (sesuai jurnal)")
    d1, d2, d3 = st.columns(3)
    with d1: st.markdown(metric_card(len(X_train), "Data Training (80%)", "blue"), unsafe_allow_html=True)
    with d2: st.markdown(metric_card(len(X_test),  "Data Testing (20%)",  "blue"), unsafe_allow_html=True)
    with d3: st.markdown(metric_card(len(X_train)+len(X_test), "Total Data Latih", "blue"), unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2 = st.tabs(["📐 Gaussian Naive Bayes", "📐 Bernoulli Naive Bayes"])

    for tab, y_pred, nama in zip([tab1, tab2], [y_pred_gnb, y_pred_bnb], ["Gaussian NB", "Bernoulli NB"]):
        with tab:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            m1,m2,m3,m4 = st.columns(4)
            with m1: st.markdown(metric_card(f"{acc*100:.2f}%",  "Accuracy",  "blue"),  unsafe_allow_html=True)
            with m2: st.markdown(metric_card(f"{prec:.2f}",      "Precision", "green"), unsafe_allow_html=True)
            with m3: st.markdown(metric_card(f"{rec:.2f}",       "Recall",    "yellow"),unsafe_allow_html=True)
            with m4: st.markdown(metric_card(f"{f1:.2f}",        "F1-Score",  "blue"),  unsafe_allow_html=True)

            st.markdown("---")
            section(f"Classification Report — {nama}")
            report = classification_report(y_test, y_pred,
                                            target_names=le.classes_, output_dict=True)
            df_rep = pd.DataFrame(report).transpose().round(3)
            st.dataframe(df_rep.style.background_gradient(cmap="Blues", subset=["precision","recall","f1-score"]),
                         use_container_width=True)

            section("Rumus Naive Bayes (Teorema Bayes)")
            st.latex(r"P(H \mid X) = \frac{P(X \mid H) \cdot P(H)}{P(X)}")
            st.markdown("""
            | Simbol | Keterangan |
            |--------|-----------|
            | **X** | Data barang yang belum diketahui kelasnya |
            | **H** | Hipotesis (kelas: LAYAK / KURANG LAYAK / TIDAK LAYAK) |
            | **P(H\|X)** | Probabilitas kelas setelah melihat data X (posterior) |
            | **P(X\|H)** | Likelihood — probabilitas fitur X jika kelas H benar |
            | **P(H)** | Prior — probabilitas awal kelas H |
            | **P(X)** | Probabilitas total data X |
            """)


# ═══════════════════════════════════════════════════════════
# PAGE: CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════
elif menu == "📊 Confusion Matrix":
    st.markdown("<h2 style='color:#1F4E79;'>📊 Confusion Matrix</h2>", unsafe_allow_html=True)
    st.markdown("Visualisasi performa prediksi model — sesuai metodologi jurnal Iskandar Madani (2025).")
    st.markdown("---")

    model_choice = st.selectbox("Pilih model:", ["Gaussian Naive Bayes", "Bernoulli Naive Bayes"])
    y_pred = y_pred_gnb if "Gaussian" in model_choice else y_pred_bnb

    cm = confusion_matrix(y_test, y_pred)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    linewidths=0.5, linecolor='white',
                    annot_kws={"size":14, "weight":"bold"})
        ax.set_title(f"Confusion Matrix — {model_choice}", fontweight='bold', fontsize=13, pad=15)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        plt.xticks(rotation=30, ha='right'); plt.yticks(rotation=0)
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        section("Keterangan Confusion Matrix")
        st.markdown("""
        | Istilah | Arti |
        |---------|------|
        | **TP** (True Positive) | Prediksi benar pada kelas positif |
        | **TN** (True Negative) | Prediksi benar pada kelas negatif |
        | **FP** (False Positive) | Salah diprediksi sebagai positif |
        | **FN** (False Negative) | Salah diprediksi sebagai negatif |
        """)
        st.markdown("---")
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
        st.markdown("---")
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
        st.success(f"**Accuracy : {acc*100:.2f}%**")
        st.info(f"**Precision: {prec:.4f}**")
        st.warning(f"**Recall   : {rec:.4f}**")

    section("Perbandingan Kedua Model")
    rows = []
    for nm, yp in [("Gaussian NB", y_pred_gnb), ("Bernoulli NB", y_pred_bnb)]:
        rows.append({
            "Model"    : nm,
            "Accuracy" : f"{accuracy_score(y_test, yp)*100:.2f}%",
            "Precision": f"{precision_score(y_test, yp, average='weighted', zero_division=0):.4f}",
            "Recall"   : f"{recall_score(y_test, yp, average='weighted', zero_division=0):.4f}",
            "F1-Score" : f"{f1_score(y_test, yp, average='weighted', zero_division=0):.4f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE: PREDIKSI MANUAL
# ═══════════════════════════════════════════════════════════
elif menu == "🔮 Prediksi Manual":
    st.markdown("<h2 style='color:#1F4E79;'>🔮 Prediksi Kondisi Barang</h2>", unsafe_allow_html=True)
    st.markdown("Masukkan data barang untuk memprediksi kelayakannya menggunakan model Naive Bayes yang sudah dilatih.")
    st.markdown("---")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        section("Input Data Barang")
        nama_input = st.text_input("Nama Barang", placeholder="Contoh: Laptop Dell")
        model_sel  = st.selectbox("Pilih Model Prediksi", ["Gaussian Naive Bayes", "Bernoulli Naive Bayes"])

        st.markdown("**Fitur Numerik:**")
        tahun  = st.number_input("Tahun Pengadaan",     min_value=2015, max_value=2030, value=2023, step=1)
        frek   = st.slider("Frekuensi Pemakaian (1=Jarang, 5=Sangat Sering)", 1, 5, 3)
        umur   = st.slider("Umur Barang (tahun)",       0, 10, 2)
        fisik  = st.slider("Kondisi Fisik (1=Rusak, 5=Sangat Baik)", 1, 5, 4)
        lengkap= st.slider("Kelengkapan Aksesori (1=Tidak Lengkap, 5=Lengkap)", 1, 5, 4)

        predict_btn = st.button("🔮 Prediksi Sekarang", use_container_width=True, type="primary")

    with col_result:
        section("Hasil Prediksi")
        if predict_btn:
            input_data = np.array([[tahun, frek, umur, fisik, lengkap]])
            model      = gnb if "Gaussian" in model_sel else bnb
            pred_enc   = model.predict(input_data)[0]
            pred_label = le.inverse_transform([pred_enc])[0]
            pred_proba = model.predict_proba(input_data)[0]

            st.markdown(f"### Barang: **{nama_input if nama_input else 'Tidak disebutkan'}**")
            st.markdown(f"**Hasil Prediksi:** {badge(pred_label)}", unsafe_allow_html=True)
            st.markdown("---")

            # Probabilitas per kelas
            section("Probabilitas per Kelas")
            prob_df = pd.DataFrame({
                "Kelas"       : le.classes_,
                "Probabilitas": [f"{p*100:.2f}%" for p in pred_proba],
                "Nilai"       : pred_proba
            })

            fig_prob, ax_prob = plt.subplots(figsize=(6, 3))
            bar_colors = [COLOR_MAP.get(c, "#999") for c in le.classes_]
            bars = ax_prob.barh(le.classes_, pred_proba, color=bar_colors, edgecolor='white')
            for bar, p in zip(bars, pred_proba):
                ax_prob.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                             f"{p*100:.1f}%", va='center', fontweight='bold')
            ax_prob.set_xlim(0, 1.15)
            ax_prob.set_xlabel("Probabilitas"); ax_prob.set_title("P(Kelas | Fitur)")
            ax_prob.set_facecolor("#F8F9FA"); fig_prob.patch.set_facecolor("white")
            ax_prob.spines[['top','right']].set_visible(False)
            st.pyplot(fig_prob, use_container_width=True)

            # Rekomendasi
            st.markdown("---")
            section("Rekomendasi")
            if pred_label == "LAYAK":
                st.success("✅ Barang dalam kondisi baik. **Tidak perlu tindakan khusus.** Lanjutkan pemantauan rutin.")
            elif pred_label == "KURANG LAYAK":
                st.warning("⚠️ Barang perlu perhatian. **Lakukan perawatan atau perbaikan minor** sebelum kondisi memburuk.")
            else:
                st.error("❌ Barang tidak layak pakai. **Segera usulkan penggantian atau perbaikan besar.** Prioritas tinggi!")

            # Ringkasan input
            st.markdown("---")
            section("Ringkasan Input")
            ringkasan = {
                "Tahun Pengadaan"    : tahun,
                "Frekuensi Pemakaian": frek,
                "Umur Barang (thn)"  : umur,
                "Kondisi Fisik"      : fisik,
                "Kelengkapan"        : lengkap,
                "Model Digunakan"    : model_sel,
                "Prediksi"           : pred_label,
            }
            st.dataframe(pd.DataFrame.from_dict(ringkasan, orient='index', columns=["Nilai"]),
                         use_container_width=True)
        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; color:#999;'>
                <h3>⬅️ Isi form dan klik tombol<br><b>Prediksi Sekarang</b></h3>
            </div>
            """, unsafe_allow_html=True)