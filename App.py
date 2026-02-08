import streamlit as st
import numpy as np
import joblib
import librosa
import parselmouth
from parselmouth.praat import call
from audiorecorder import audiorecorder
import io

# ──────────────────────────────────────────────
# Page Config & Styling
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Parkinson's Voice Risk Screening",
    layout="centered",
    page_icon="🧠",
)

st.markdown(
    """
    <style>
    .main { background-color: #0f172a; }
    h1, h2, h3 { color: #e5e7eb; }
    p, div { color: #cbd5e1; }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Load Model & Scaler (cached so they load once)
# ──────────────────────────────────────────────

@st.cache_resource
def load_model():
    model = joblib.load("pd_model_clean_v2.pkl")
    scaler = joblib.load("scaler_v2.pkl")
    return model, scaler

model, scaler = load_model()

# ──────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────

def extract_features(path: str) -> list:
    """Extract 21 acoustic features from a WAV file using Praat + Librosa."""
    snd = parselmouth.Sound(path)

    # --- Pitch ---
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    fo  = call(pitch, "Get mean",    0, 0, "Hertz")
    fhi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    flo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

    # --- Jitter ---
    pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter_local = call(pp, "Get jitter (local)",            0, 0, 0.0001, 0.02, 1.3)
    jitter_abs   = call(pp, "Get jitter (local, absolute)",  0, 0, 0.0001, 0.02, 1.3)
    rap          = call(pp, "Get jitter (rap)",              0, 0, 0.0001, 0.02, 1.3)
    ppq          = call(pp, "Get jitter (ppq5)",             0, 0, 0.0001, 0.02, 1.3)
    ddp          = rap * 3

    # --- Shimmer ---
    shimmer_local = call([snd, pp], "Get shimmer (local)",    0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_db    = call([snd, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3          = call([snd, pp], "Get shimmer (apq3)",     0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5          = call([snd, pp], "Get shimmer (apq5)",     0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq           = call([snd, pp], "Get shimmer (apq11)",    0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda           = apq3 * 3

    # --- HNR ---
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    # --- Nonlinear / proxy features (Librosa) ---
    y, sr = librosa.load(path, sr=16000)
    zcr     = librosa.feature.zero_crossing_rate(y).mean()
    spread1 = -abs(zcr)
    spread2 = -abs(np.var(y))
    dfa     = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)) / 10
    rpde    = np.std(y)
    d2      = np.var(y)
    ppe     = np.mean(np.abs(y))

    return [
        fo, fhi, flo,
        jitter_local, jitter_abs, rap, ppq, ddp,
        shimmer_local, shimmer_db, apq3, apq5, apq, dda,
        hnr, rpde, d2, dfa, spread1, spread2, ppe,
    ]

# ──────────────────────────────────────────────
# Results Display
# ──────────────────────────────────────────────

def show_results(file_path: str):
    """Run the ML pipeline and render the results section."""
    with st.spinner("Extracting acoustic features…"):
        features = extract_features(file_path)

    with st.spinner("Running model prediction…"):
        X_scaled = scaler.transform([features])
        pred     = model.predict(X_scaled)[0]
        pd_risk  = model.predict_proba(X_scaled)[0][1]

    risk_percent = int(pd_risk * 100)

    # --- Gauge ---
    st.subheader("📈 Risk Gauge")
    st.progress(risk_percent)

    if pred == 1:
        st.markdown(f"🔴 **Parkinson's Risk Level:** {risk_percent}%")
    else:
        st.markdown(f"🟢 **Healthy Voice Confidence:** {100 - risk_percent}%")

    # --- Detailed result ---
    st.subheader("📊 Screening Result")

    if pred == 1:
        st.error("⚠️ Elevated Risk Detected")
        st.metric("Estimated Risk Probability", f"{pd_risk * 100:.1f}%")
        st.markdown(
            "🟡 This does **not** mean the person has Parkinson's. "
            "It suggests they *should consider further clinical evaluation* if possible."
        )
    else:
        st.success("✅ Low Risk Detected")
        st.metric("Estimated Risk Probability", f"{pd_risk * 100:.1f}%")
        st.markdown(
            "🟢 No strong Parkinson's indicators found in this sample. "
            "Still, this is **not a medical diagnosis**."
        )

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.title("🧠 Parkinson's Voice Risk Screening — Prototype")
st.markdown(
    """
    **Purpose:** This is a *risk screening* tool — not a diagnosis system.  
    It analyzes voice patterns linked to Parkinson's and flags **elevated risk**
    for people who may not have access to regular medical checkups.

    ⚠️ *Educational & research prototype only.*
    """
)

st.divider()

# ──────────────────────────────────────────────
# Input: Record or Upload (tabs)
# ──────────────────────────────────────────────

tab_record, tab_upload = st.tabs(["🎙️ Record Voice", "📤 Upload File"])

# --- Record tab ---
with tab_record:
    st.caption("Click to record a sustained 'Ahhh' sound for 5–15 seconds.")
    audio = audiorecorder("Click to Record", "Click to Stop")

    if audio is not None and len(audio) > 0:
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        wav_bytes = buffer.getvalue()

        st.audio(wav_bytes, format="audio/wav")

        if st.button("🔍 Analyze Recording", key="analyze_rec"):
            with open("recorded.wav", "wb") as f:
                f.write(wav_bytes)
            show_results("recorded.wav")

# --- Upload tab ---
with tab_upload:
    uploaded_file = st.file_uploader("Upload a .wav recording", type=["wav"])
    st.caption("Recommended: 5–15 seconds of sustained vowel sound.")

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("🔍 Analyze Upload", key="analyze_upl"):
            with open("temp.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            show_results("temp.wav")

# ──────────────────────────────────────────────
# How It Works + Technical Details
# ──────────────────────────────────────────────

st.divider()

st.subheader("🔍 How It Works")
st.markdown(
    """
    1. **Extract** — pitch, jitter, shimmer, and harmonic features are pulled from the audio.  
    2. **Scale** — features are standardized using trained statistics.  
    3. **Predict** — a trained ML classifier estimates a **risk probability**.  
    """
)

with st.expander("🔬 Technical & Research Foundations"):
    st.markdown(
        """
### 🧠 Machine Learning Pipeline

**Signal Processing:** Voice recordings are processed using *Praat / Parselmouth*
and *Librosa* to extract acoustic biomarkers.

**Feature Engineering:** Clinically relevant features include fundamental
frequency (F0), jitter & shimmer (voice instability), harmonic-to-noise ratio
(HNR), and nonlinear spectral measures.

**Modeling:** Features are standardized and passed into a trained supervised ML
classifier optimized for small biomedical datasets.

**Validation Mindset:** We treat this as a risk screening tool, not a diagnostic
system. The model is evaluated for *generalization*, not memorization.

---

### ⚖️ Ethics & Purpose

This prototype is designed for early risk screening in underserved populations
and as a low-cost, accessible healthcare support tool. It does **not replace
clinicians** and should only be used as a decision-support tool.

---

### 🧪 Research Context

This project draws inspiration from published work on voice biomarkers in
Parkinson's Disease, acoustic instability as a neurological signal, and
ML-based screening for neurodegenerative disorders.

**Future directions:** larger datasets, speaker-independent validation, hybrid
deep learning architectures, and mobile deployment.
        """
    )

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.divider()
st.caption(
    "**Built by:** Team Youel Fissha & Ahla Usman · "
    "AI Voice-Based Parkinson's Risk Screening · "
    "Blue Ocean Entrepreneurship Competition"
)