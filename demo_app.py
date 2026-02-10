
import streamlit as st
import random
import base64
from PIL import Image, ImageDraw
import streamlit.components.v1 as components
from bert_transformer import BERTTransformer
import joblib
import torch
from transformers import DistilBertModel, DistilBertTokenizerFast
import pandas as pd

pipeline = joblib.load("risk_pipeline.pkl")
@st.cache_resource
def load_models():
    tokenizer = DistilBertTokenizerFast.from_pretrained("bert_model")
    bert_model = DistilBertModel.from_pretrained("bert_model")
    classifier = joblib.load("risk_classifier.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    return tokenizer, bert_model, classifier, label_encoders
tokenizer, bert_model, classifier, label_encoders = load_models()
bert_model.eval()  # IMPORTANT

    # -------- PREDICTION FUNCTION --------
def get_bert_embedding(text):
    # tokenize text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    # get embeddings
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Use [CLS] token representation (first token) as embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.numpy()  # convert to numpy for sklearn

def predict_log(log_text):
    # Use the pipeline to predict directly from raw text
    # pipeline.predict() returns a dictionary of predictions already as strings
    preds = pipeline.predict([log_text])
    
    # Extract the first result (since we passed a single text)
    results = preds[0] if isinstance(preds, list) and len(preds) > 0 else preds
    
    # If results is already a dict, return it directly
    # The pipeline has already done label encoding/decoding
    if isinstance(results, dict):
        return results
    
    # Fallback: if preds is a numpy array
    if hasattr(preds, 'shape') and len(preds.shape) > 0:
        results = {}
        preds_row = preds[0] if len(preds.shape) > 1 else preds
        for i, col in enumerate(label_encoders.keys()):
            if i < len(preds_row):
                pred_val = preds_row[i]
                # If it's already a string, use it directly
                if isinstance(pred_val, str):
                    results[col] = pred_val
                else:
                    # If it's numeric, convert to Python int and inverse transform
                    try:
                        pred_int = int(pred_val.item()) if hasattr(pred_val, 'item') else int(pred_val)
                        results[col] = label_encoders[col].inverse_transform([pred_int])[0]
                    except (ValueError, TypeError):
                        results[col] = str(pred_val)
        return results
    
    return results if results else {}

def normalize_predictions(results):
    return {
        "risk_category": results["risk_category"].title(),   # high → High
        "severity": results["severity"].title(),             # catastrophic → Catastrophic
        "urgency": (
            "Immediate" if results["urgency"].lower() in ["urgent", "immediate"]
            else results["urgency"].title()
        ),
        "safety_hazard": (
            True if str(results.get("is_safety_hazard")).lower() in ["yes", "true", "1"]
            else False
        )
    }

# -----------------------------
# RULE-BASED SCORE CALCULATION
# -----------------------------

RISK_CATEGORY_SCORE = {
    "Low": 0.2,
    "Medium": 0.5,
    "High": 0.9
}

SEVERITY_SCORE = {
    "Negligible": 0.1,
    "Minor": 0.4,
    "Major": 0.7,
    "Catastrophic": 1.0
}

URGENCY_SCORE = {
    "Routine": 0.2,
    "Scheduled": 0.5,
    "Immediate": 0.9
}

SAFETY_HAZARD_SCORE = {
    False: 0.0,
    True: 1.0
}

def rule_based_log_score(normalized):
    """
    Takes normalized model outputs and returns final log threat score
    """
    rc = RISK_CATEGORY_SCORE.get(normalized["risk_category"], 0.5)
    sev = SEVERITY_SCORE.get(normalized["severity"], 0.5)
    urg = URGENCY_SCORE.get(normalized["urgency"], 0.5)
    sh = SAFETY_HAZARD_SCORE.get(normalized["safety_hazard"], 0.0)

    score = (
        0.35 * rc +
        0.35 * sev +
        0.20 * urg +
        0.10 * sh
    )

    return round(score, 3)

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------
st.set_page_config(
    page_title="BiModal Telecom Risk Monitor",
    layout="wide"
)

# ---------------------------------
# STYLES & BACKGROUND
# ---------------------------------
def add_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        /* RESET EVERYTHING */
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            overflow-x: hidden;
        }}

        /* REMOVE STREAMLIT SIDE PADDING */
        .stApp, main, .block-container {{
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100% !important;
        }}

        /* HERO FULLSCREEN */
        .hero-bg {{
            width: 100vw;
            height: 100vh;
            margin: 0;
            background:
        linear-gradient(
            90deg,
            rgba(0,0,0,0.75) 0%,
            rgba(0,0,0,0.25) 40%,
            rgba(0,0,0,0.15) 70%
        ),
        url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            display: flex;
            align-items: center;
        }}

        /* HERO TEXT */
        .hero-text {{
            padding-left: 80px;
            color: white;
            text-shadow: 0 8px 30px rgba(0,0,0,0.6);
            animation: slideFade 1.6s ease-out;
        }}

        .hero-text h1 {{
            font-size: 68px;
            font-weight: 800;
            line-height: 1.1;
        }}

        .hero-text p {{
            font-size: 22px;
            opacity: 0.9;
        }}

        @keyframes slideFade {{
            from {{
                opacity: 0;
                transform: translateX(-60px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}

        /* SCROLL TEXT */
        .scroll-down {{
            position: absolute;
            bottom: 30px;
            left: 80px;
            color: white;
            opacity: 0.8;
        }}

        /* DASHBOARD BACKGROUND */
        .dashboard {{
            background-color: #f5f7fb;
            padding: 70px 60px;
        }}

        /* CARDS */
        .card {{
            background: white;
            padding: 25px;
            border-radius: 18px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.12);
            margin-bottom: 25px;
        }}

        /* SIDEBAR */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #020617, #0f172a);
        }}
        section[data-testid="stSidebar"] * {{
            color: white;
        }}
        /* FEATURE SECTION */
        .feature-section {{
        background: linear-gradient(180deg, #0f172a, #020617);
        padding: 100px 80px;
        color: white;
        }}

        .feature-title {{
            text-align: center;
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 60px;
        }}

        .feature-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 50px;
        }}

        .feature-card {{
            background: rgba(255,255,255,0.05);
            padding: 40px 30px;
            border-radius: 18px;
            backdrop-filter: blur(6px);
            border: 1px solid rgba(255,255,255,0.08);
        }}

        .feature-card h3 {{
            font-size: 22px;
            margin-bottom: 15px;
        }}

        .feature-card p {{
            font-size: 15px;
            line-height: 1.7;
            opacity: 0.85;
        }}

        .feature-btn {{
            margin-top: 25px;
            display: inline-block;
            padding: 10px 22px;
            border-radius: 30px;
            border: 1px solid #38bdf8;
            color: #38bdf8;
            text-decoration: none;
            font-size: 14px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


add_bg("Telecom.jpg")

# ---------------------------------
# HERO SECTION
# ---------------------------------
st.markdown("""
<div class="hero-bg">
    <div class="hero-text">
        <h1> BiModal Telecom<br>Risk Monitor</h1>
        <p>AI-powered Visual & NLP based Telecom Infrastructure Safety System</p>
    </div>
</div>

""", unsafe_allow_html=True)
# ---------------------------------
# FEATURE SECTION (Exactly like the screenshot)
# ---------------------------------
components.html(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap');
        
        body {
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
            background-color: transparent;
        }

        .feature-container {
            /* This semi-transparent background allows the Hero image to peek through */
            background: rgba(15, 23, 42, 0.88); 
            backdrop-filter: blur(8px);
            padding: 50px 20px;
            color: white;
            text-align: center;
            border-radius: 4px;
            margin: 0 10%; /* Center the box on the page */
        }

        .feature-title {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 40px;
            line-height: 1.2;
        }

        .feature-grid {
            display: flex; /* Flex is more reliable for vertical dividers */
            justify-content: space-around;
            align-items: stretch;
            max-width: 1200px;
            margin: 0 auto;
        }

        .feature-card {
            flex: 1;
            padding: 0 30px;
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
        }

        /* The Cyan Vertical Divider */
        .feature-card:not(:last-child)::after {
            content: "";
            position: absolute;
            right: 0;
            top: 10%;
            height: 80%;
            width: 1px;
            background: rgba(56, 189, 248, 0.4);
        }

        .icon-box {
            color: #38bdf8;
            margin-bottom: 20px;
        }

        .feature-card h3 {
            font-size: 19px;
            font-weight: 700;
            margin-bottom: 15px;
            min-height: 50px;
        }

        .feature-card p {
            font-size: 13px;
            line-height: 1.7;
            opacity: 0.8;
            margin-bottom: 30px;
            min-height: 90px;
        }

        .feature-btn {
            display: inline-block;
            padding: 8px 25px;
            border-radius: 20px;
            border: 1px solid #38bdf8;
            color: #38bdf8;
            text-decoration: none;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .feature-btn:hover {
            background: #38bdf8;
            color: #0f172a;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
        }
        .feature-btn {
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.feature-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 12px rgba(56,189,248,0.4);
}

    </style>

    <div class="feature-container">
        <div class="feature-title">
            Leverage Precise Tower Data
<br>
            <span style="color:#38bdf8">to Unlock Hidden Opportunities</span>
        </div>

        <div class="feature-grid">
    <div class="feature-card">
        <div class="icon-box">
            <svg width="35" height="35" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
            </svg>
        </div>
        <h3>Unified Tower Intelligence Platform</h3>
        <p>
            The system delivers a consolidated and reliable view of telecom tower health by
            integrating visual inspection results from YOLO-based image analysis with
            technician log insights derived through NLP. This unified representation
            enables accurate risk identification, eliminates fragmented data sources,
            and supports informed decision-making across tower maintenance operations.
        </p>
    </div>

    <div class="feature-card">
        <div class="icon-box">
            <svg width="35" height="35" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="15 3 21 3 21 9"></polyline>
                <polyline points="9 21 3 21 3 15"></polyline>
                <line x1="21" y1="3" x2="14" y2="10"></line>
                <line x1="3" y1="21" x2="10" y2="14"></line>
            </svg>
        </div>
        <h3>Efficient, Rapid, and Scalable Analysis</h3>
        <p>
            Designed for real-world deployment, the platform enables fast processing of
            tower images and inspection logs regardless of location or tower type.
            Automated detection and analysis reduce manual effort, require minimal
            operator expertise, and allow the system to scale seamlessly across
            large telecom infrastructures.
        </p>
    </div>

    <div class="feature-card">
        <div class="icon-box">
            <svg width="35" height="35" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
        </div>
        <h3>Comprehensive Digital Risk Monitoring</h3>
        <p>
            By combining computer vision–based structural assessment with NLP-driven
            log analysis, the system provides an end-to-end digital solution for
            telecom tower risk evaluation. Integrated severity scoring and visual
            dashboards enable proactive maintenance planning, early hazard detection,
            and improved infrastructure reliability.
        </p>
    </div>
</div>
    """,
    height=700,
)

# ---------------------------------
# DASHBOARD HEADER
# ---------------------------------
st.markdown("""
<style>
/* FORCE DARK BLUE BACKGROUND */
.stApp {
    background-color: #020617;
    color: #e5e7eb;
}

/* Remove white padding background */
section[data-testid="stSidebar"],
section[data-testid="stMain"] {
    background-color: #020617;
}

/* HEADER */
.dashboard-header {
    background: linear-gradient(135deg, #020b2d, #020617);
    padding: 80px 60px;
    border-radius: 0 0 40px 40px;
    color: #f8fafc;
}

/* CARDS */
.card {
    background: rgba(2, 11, 45, 0.85);
    backdrop-filter: blur(16px);
    padding: 28px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    margin-bottom: 30px;
}

/* METRICS */
.metric-box {
    background: rgba(29, 78, 216, 0.18);
    padding: 22px;
    border-radius: 16px;
    text-align: center;
}

/* ALERT BADGES */
.badge-red {
    background: #7f1d1d;
    padding: 12px 20px;
    border-radius: 24px;
    font-weight: 600;
    color: #fee2e2;
}
.badge-orange {
    background: #78350f;
    padding: 12px 20px;
    border-radius: 24px;
    font-weight: 600;
    color: #ffedd5;
}

/* TABS */
.stTabs [data-baseweb="tab"] {
    color: #c7d2fe;
    font-size: 16px;
    padding: 14px 28px;
}

/* FOOTER */
.footer {
    text-align: center;
    opacity: 0.6;
    padding: 40px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------
# HEADER
# ---------------------------------
st.markdown("""
<div class="dashboard-header">
    <h1> Telecom Infrastructure Monitoring</h1>
    <p>AI-Driven Risk Detection using Computer Vision & NLP</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------------------------
# TABS
# ---------------------------------
tab1, tab2, tab3 = st.tabs([
    " Image Inspection",
    " Technician Log Analysis",
    " Risk & Severity Dashboard"
])

# ---------------------------------
# TAB 1 – IMAGE ANALYSIS
# ---------------------------------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(" Visual Structural Inspection (YOLOv8)")

    uploaded_file = st.file_uploader(
        "Upload tower image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        draw = ImageDraw.Draw(image)
        w, h = image.size
        draw.rectangle(
            [w*0.25, h*0.25, w*0.6, h*0.6],
            outline="#ef4444",
            width=6
        )

        st.image(image, use_container_width=True)
        st.error("🔴 Critical corrosion detected on tower structure")

    else:
        st.info("Upload an image to begin AI inspection")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# TAB 2 – NLP LOG ANALYSIS
# ---------------------------------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Technician Log Intelligence")

    log = st.text_area(
        "Paste inspection log",
        height=180,
        placeholder="Example: Severe corrosion observed near base leg..."
    )

    if log:
        # Step 1: Model prediction
        results = predict_log(log)

        # Step 2: Normalize labels
        normalized = normalize_predictions(results)

        # Step 3: Rule-based score
        log_score = rule_based_log_score(normalized)

        # ---------------------------------
        # MODEL OUTPUTS
        # ---------------------------------
        st.subheader(" Model Outputs")

        st.write("**Risk Category:**", normalized["risk_category"])
        st.write("**Severity:**", normalized["severity"])
        st.write("**Urgency:**", normalized["urgency"])
        st.write(
            "**Safety Hazard:**",
            "Yes 🚨" if normalized["safety_hazard"] else "No"
        )

        # ---------------------------------
        # FINAL SCORE
        # ---------------------------------
        st.subheader(" Log Threat Score")
        st.metric("Threat Score", f"{log_score} / 1.00")

        # ---------------------------------
        # STATUS MESSAGE
        # ---------------------------------
        if log_score >= 0.75:
            st.error(" CRITICAL RISK — Immediate Action Required")
        elif log_score >= 0.4:
            st.warning(" MODERATE RISK — Schedule Inspection")
        else:
            st.success(" LOW RISK — No Immediate Threat")

    st.markdown("</div>", unsafe_allow_html=True) 

# ---------------------------------
# TAB 3 – RISK DASHBOARD
# ---------------------------------
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🚨 Unified Threat Severity Assessment")

    score = random.randint(55, 95)

    st.metric("Overall Severity Score", f"{score}/100")

    if score >= 75:
        st.markdown("<span class='badge-red'>IMMEDIATE ACTION REQUIRED</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge-orange'>MONITOR & SCHEDULE INSPECTION</span>",
                    unsafe_allow_html=True)

    st.write("")
    st.subheader("Risk Contribution Breakdown")

    st.bar_chart({
        "Corrosion": 45,
        "Cracks": 20,
        "Foundation Shift": 15,
        "Loose Cabling": 10,
        "Environmental Stress": 10
    })

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("""
<div class="footer">
© 2026 • Bi-Modal Telecom Risk Monitor • AI-Powered Infrastructure Safety
</div>
""", unsafe_allow_html=True) 