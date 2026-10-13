import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

# ── Load model from HuggingFace Hub ──────────────────────────────────
MODEL_ID = "24f2003563/smart-mcq-solver-deberta"  # your hub repo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Smart MCQ Solver", page_icon="🧠", layout="centered")

# ── Custom theme (dark navy + blue accent) ───────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a0e17; }

    .badge {
        display: block;
        margin: 8px auto 20px auto;
        width: fit-content;
        padding: 6px 18px;
        border: 1px solid #3b5bdb;
        border-radius: 999px;
        color: #6c8dff;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .app-title {
        text-align: center;
        font-size: 44px;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 8px 0;
    }
    .app-subtitle {
        text-align: center;
        color: #8a93a6;
        font-size: 15px;
        margin-bottom: 32px;
    }

    label, .stTextArea label, .stTextInput label {
        color: #b6bccb !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .stTextArea textarea, .stTextInput input {
        background-color: #0f1420 !important;
        border: 1px solid #232a3d !important;
        border-radius: 10px !important;
        color: #e7e9ee !important;
    }
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: #57607a !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #3b5bdb !important;
        box-shadow: 0 0 0 1px #3b5bdb !important;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(180deg, #4c6fff, #3757e0);
        color: white;
        font-weight: 700;
        font-size: 16px;
        padding: 14px 0;
        border: none;
        border-radius: 10px;
        box-shadow: 0 8px 24px -8px rgba(59, 91, 219, 0.6);
        transition: filter 0.15s ease;
    }
    div.stButton > button:hover {
        filter: brightness(1.08);
        color: white;
    }

    .result-card {
        background-color: #0f1420;
        border: 1px solid #232a3d;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    .result-rank {
        color: #6c8dff;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.05em;
    }
    .result-text {
        color: #e7e9ee;
        font-size: 16px;
        margin: 4px 0;
    }
    .result-score {
        color: #8a93a6;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    config = PeftConfig.from_pretrained(MODEL_ID)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path, num_labels=2
    )
    model = PeftModel.from_pretrained(base_model, MODEL_ID)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return tokenizer, model


OPTIONS = ["A", "B", "C", "D", "E"]


def predict(question, options):
    scores = []
    for opt_text in options:
        text = f"Question: {question} Candidate Answer: {opt_text}"
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, padding="max_length", max_length=256
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits.float()
            prob = F.softmax(logits, dim=-1)[0, 1].item()
        scores.append(prob)
    ranked = sorted(zip(OPTIONS, options, scores), key=lambda x: x[2], reverse=True)
    return ranked[:3]


# ── Header ────────────────────────────────────────────────────────────
st.markdown('<div class="badge">DeBERTa Multiple-Choice AI</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">Smart MCQ Solver</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Enter a question prompt and 5 options to generate ranked predictions.</div>',
    unsafe_allow_html=True,
)

with st.spinner("Loading model... (first run can take a minute)"):
    tokenizer, model = load_model()

# ── Form ──────────────────────────────────────────────────────────────
question = st.text_area("Question Prompt", placeholder="Enter your question here...", height=100)

col1, col2 = st.columns(2)
with col1:
    option_a = st.text_input("Option A", placeholder="Enter option A")
    option_c = st.text_input("Option C", placeholder="Enter option C")
    option_e = st.text_input("Option E", placeholder="Enter option E")
with col2:
    option_b = st.text_input("Option B", placeholder="Enter option B")
    option_d = st.text_input("Option D", placeholder="Enter option D")

if st.button("Predict Best Answers"):
    options = [option_a, option_b, option_c, option_d, option_e]
    if not question.strip() or any(not o.strip() for o in options):
        st.warning("Please fill in the question and all 5 options.")
    else:
        with st.spinner("Scoring options..."):
            top3 = predict(question, options)

        st.write("")
        for rank, (letter, text, score) in enumerate(top3, 1):
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-rank">#{rank} · OPTION {letter}</div>
                    <div class="result-text">{text}</div>
                    <div class="result-score">Confidence: {score:.3f}</div>
                </div>
            """, unsafe_allow_html=True)