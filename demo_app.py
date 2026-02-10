import streamlit as st
import requests  # ADD THIS IMPORT
import random
import base64
from PIL import Image, ImageDraw
import streamlit.components.v1 as components
from bert_transformer import BERTTransformer
import joblib
import torch
from transformers import DistilBertModel, DistilBertTokenizerFast
import pandas as pd

# ---------------------------------
# ADD MISSING FUNCTIONS FIRST
# ---------------------------------

def call_yolo_api(image_file):
    """
    Call your LOCAL YOLO+ResNet API
    Returns: JSON with equipment counts and corrosion level
    """
    try:
        # Send image to your local API
        files = {"file": image_file.getvalue()}
        
        # Call your API running on localhost:8000
        response = requests.post(
            "http://localhost:8000/analyze",
            files=files,
            timeout=30  # Give time for ML models
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"❌ Cannot connect to YOLO API: {str(e)}")
        st.info("💡 Make sure your API is running: python main.py")
        return None
#image score calc -dynamic
def calculate_image_risk(api_results):
    """
    Convert YOLO+ResNet API results to a 0-1 risk score
    FIXED VERSION - handles your corrosion labels correctly
    """
    if not api_results:
        print("DEBUG: No API results, returning 0.5")
        return 0.5
    
    try:
        # DEBUG: Print what we're receiving
        print("=" * 50)
        print("DEBUG calculate_image_risk:")
        print(f"Full API results: {api_results}")
        
        # Get corrosion level from your API response
        condition = api_results.get("condition_analysis", {})
        corrosion = str(condition.get("corrosion_level", "unknown")).strip()
        confidence = condition.get("confidence", 0.5)
        
        print(f"Corrosion raw: '{corrosion}'")
        print(f"Confidence: {confidence}")
        
        # Normalize corrosion string (handle your specific labels)
        corrosion_lower = corrosion.lower()
        
        # Your corrosion labels are: "High-corrosion", "Medium-corrosion", "Low-corrosion"
        # Map them to risk scores
        corrosion_map = {
            "high-corrosion": 0.9,
            "high": 0.9,
            "severe": 1.0,
            "medium-corrosion": 0.6,
            "medium": 0.6,
            "moderate": 0.6,
            "low-corrosion": 0.3,
            "low": 0.3,
            "none": 0.1,
            "unknown": 0.5
        }
        
        # Find matching key (handles partial matches)
        base_score = 0.5  # Default
        for key, score in corrosion_map.items():
            if key in corrosion_lower:
                base_score = score
                print(f"Matched '{key}' -> score {score}")
                break
        
        print(f"Base corrosion score: {base_score}")
        
        # Adjust with confidence
        adjusted_score = base_score * (0.5 + 0.5 * confidence)
        print(f"Adjusted with confidence: {adjusted_score}")
        
        # Add equipment risk
        equipment = api_results.get("equipment_analysis", {})
        print(f"Equipment: {equipment}")
        
        equipment_risk = 0.0
        if equipment:
            # Count total equipment as proxy for complexity/risk
            total_items = sum(equipment.values())
            equipment_risk = min(0.3, total_items * 0.05)  # Max 0.3 from equipment
            
            # Extra risk for damaged equipment
            damage_keywords = ["damage", "crack", "rust", "broken", "missing", "bent", "leaning"]
            for keyword in damage_keywords:
                for item in equipment.keys():
                    if keyword in str(item).lower():
                        equipment_risk += 0.1  # Extra for damage
                        print(f"Found damage keyword '{keyword}' in '{item}'")
        
        print(f"Equipment risk: {equipment_risk}")
        
        # Calculate total score
        total_score = min(1.0, adjusted_score + equipment_risk)
        print(f"Total score: {total_score}")
        print("=" * 50)
        
        return round(total_score, 2)
        
    except Exception as e:
        print(f"ERROR in calculate_image_risk: {e}")
        import traceback
        traceback.print_exc()
        return 0.5

# ---------------------------------
# YOUR EXISTING NLP CODE
# ---------------------------------
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

def calculate_log_threat_score(risk_category, severity, urgency, safety_hazard):
    """
    Compute a normalized threat score in [0, 1] from model outputs.

    Parameters accepted as strings (e.g. 'high', 'medium'), booleans, or numeric-like values.
    The function maps common categorical labels to numeric weights and combines
    them with tuned component weights to produce a single score.
    """
    def _map(val, mapping, default=0.5):
        if isinstance(val, str):
            return mapping.get(val.strip().lower(), default)
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        try:
            return float(val)
        except Exception:
            return default

    risk_map = {
        'critical': 1.0, 'critical_risk': 1.0, 'high': 0.9, 'medium': 0.6,
        'moderate': 0.6, 'low': 0.2, 'minor': 0.2, 'none': 0.0
    }
    severity_map = {
        'critical': 1.0, 'severe': 1.0, 'high': 0.9, 'major': 0.9,
        'medium': 0.6, 'moderate': 0.6, 'low': 0.2, 'minor': 0.2
    }
    urgency_map = {
        'immediate': 1.0, 'now': 1.0, 'urgent': 0.9, 'soon': 0.6,
        'routine': 0.2, 'low': 0.2, 'none': 0.0
    }

    r = _map(risk_category, risk_map, default=0.5)
    s = _map(severity, severity_map, default=0.5)
    u = _map(urgency, urgency_map, default=0.5)

    if isinstance(safety_hazard, str):
        sh = 1.0 if safety_hazard.strip().lower() in ('yes', 'true', 'y', '1') else 0.0
    else:
        sh = 1.0 if bool(safety_hazard) else 0.0

    # Component weights (sum to 1.0)
    w_r, w_s, w_u, w_sh = 0.4, 0.35, 0.2, 0.05

    score = r * w_r + s * w_s + u * w_u + sh * w_sh
    return max(0.0, min(1.0, score))

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
    "📸 Image Inspection",
    "📝 Technician Log Analysis", 
    "📊 Risk & Severity Dashboard"
])

# ---------------------------------
# TAB 1 – IMAGE ANALYSIS (UPDATED)
# ---------------------------------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Visual Structural Inspection (YOLOv8 + ResNet)")
    
    # API Status Check
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("📡 Check API Connection", type="secondary"):
            try:
                response = requests.get("http://localhost:8000/", timeout=5)
                if response.status_code == 200:
                    st.success("✅ API Connected")
                else:
                    st.warning(f"⚠️ API returned {response.status_code}")
            except Exception as e:
                st.error(f"❌ API Not Found: {e}")
                st.info("Run: python main.py in another terminal")
    
    uploaded_file = st.file_uploader(
        "Upload telecom tower image",
        type=["jpg", "png", "jpeg"]
    )
    
    if uploaded_file:
        # Show preview
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Analyze button
        if st.button("🚀 Run AI Analysis", type="primary"):
            with st.spinner("Analyzing with YOLO+ResNet..."):
                # Call YOUR YOLO+ResNet API
                results = call_yolo_api(uploaded_file)
                
                if results:
                    st.success("✅ Analysis Complete!")
                    
                    # Display equipment counts
                    equipment = results.get("equipment_analysis", {})
                    if equipment:
                        st.subheader("📊 Detected Equipment")
                        # Create columns for equipment display
                        cols = st.columns(3)
                        for i, (item, count) in enumerate(equipment.items()):
                            with cols[i % 3]:
                                st.metric(item.replace("_", " ").title(), count)
                    
                    # Display corrosion analysis
                    condition = results.get("condition_analysis", {})
                    corrosion = condition.get("corrosion_level", "unknown")
                    confidence = condition.get("confidence", 0)
                    
                    st.subheader("⚠️ Condition Assessment")
                    
                    # Color-coded corrosion alert
                    corrosion_lower = str(corrosion).lower()
                    if "high" in corrosion_lower or "severe" in corrosion_lower:
                        st.error(f"🔴 CORROSION: {corrosion.upper()} (Confidence: {confidence:.1%})")
                    elif "medium" in corrosion_lower:
                        st.warning(f"🟡 Corrosion: {corrosion} (Confidence: {confidence:.1%})")
                    else:
                        st.success(f"🟢 Corrosion: {corrosion} (Confidence: {confidence:.1%})")
                    
                    # Calculate and show risk score
                    risk_score = calculate_image_risk(results)
                    st.metric("📈 Image Risk Score", f"{risk_score:.2f}/1.0")
                    
                    # Save for Tab 3 dashboard
                    st.session_state['last_image_score'] = risk_score
                    
                    # Optional: Show raw data
                    with st.expander("📄 View Detailed Results"):
                        st.json(results)
                else:
                    st.error("Failed to get analysis results")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# TAB 2 – NLP LOG ANALYSIS
# ---------------------------------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📝 Technician Log Intelligence")

    log = st.text_area(
        "Paste inspection log",
        height=180,
        placeholder="Example: Severe corrosion observed near base leg..."
    )

    # -------- RUN MODEL --------
    if log:
        results = predict_log(log)

        risk_category = results.get("risk_category", "unknown")
        severity = results.get("severity", "medium")
        urgency = results.get("urgency", "routine")
        safety_hazard = results.get("is_safety_hazard", False)

        log_score = calculate_log_threat_score(
            risk_category, severity, urgency, safety_hazard
        )

        st.metric("Log Risk Score", f"{log_score:.2f} / 1.0")

        if log_score > 0.75:
            st.error("🔴 High Risk Log Detected")
        elif log_score > 0.5:
            st.warning("🟡 Medium Risk Log")
        else:
            st.success("🟢 Low Risk Log")
            
        # Save for Tab 3 dashboard
        st.session_state['last_nlp_score'] = log_score

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# TAB 3 – RISK DASHBOARD (UPDATED)
# ---------------------------------
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Unified Threat Severity Assessment")
    
    # Get scores (with defaults)
    image_score = st.session_state.get('last_image_score', 0.5)
    nlp_score = st.session_state.get('last_nlp_score', 0.5)
    
    # Combined score (60% image, 40% NLP)
    overall_score = (image_score * 0.6) + (nlp_score * 0.4)
    
    # Display overall score
    st.metric("Overall Risk Score", f"{overall_score:.2f}/1.0")
    
    # Progress bar visualization
    st.progress(float(overall_score))
    
    # Score breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Image Analysis Score", f"{image_score:.2f}")
    with col2:
        st.metric("Log Analysis Score", f"{nlp_score:.2f}")
    
    # Risk level and recommendations
    st.subheader("🚨 Risk Assessment")
    
    if overall_score > 0.7:
        st.error("""
        🔴 **CRITICAL RISK** - IMMEDIATE ACTION REQUIRED
        - Dispatch emergency inspection team
        - Consider temporary shutdown
        - Notify safety authorities
        """)
        st.markdown("<span class='badge-red'>IMMEDIATE ACTION REQUIRED</span>", unsafe_allow_html=True)
    elif overall_score > 0.4:
        st.warning("""
        🟡 **MODERATE RISK** - SCHEDULE INSPECTION
        - Schedule inspection within 7 days
        - Monitor daily
        - Prepare maintenance resources
        """)
        st.markdown("<span class='badge-orange'>MONITOR & SCHEDULE INSPECTION</span>", unsafe_allow_html=True)
    else:
        st.success("""
        🟢 **LOW RISK** - NORMAL OPERATION
        - Continue routine monitoring
        - Schedule next quarterly inspection
        - No immediate action needed
        """)

    st.write("")
    st.subheader("📈 Risk Contribution Breakdown")

    # Create dynamic risk breakdown
    risk_data = {
        "Visual Defects": min(50, image_score * 50),
        "Log Severity": min(30, nlp_score * 30),
        "Equipment Issues": min(15, image_score * 15),
        "Other Factors": 5
    }
    
    st.bar_chart(risk_data)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("""
<div class="footer">
© 2026 • Bi-Modal Telecom Risk Monitor • AI-Powered Infrastructure Safety
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# SIDEBAR WITH API INFO
# ---------------------------------
