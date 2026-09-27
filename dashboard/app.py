import json
from pathlib import Path

import joblib
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "ddos_model.pkl"
DATA_PATH = ROOT / "data" / "processed" / "ddos_cleaned.csv"
RESULT_PATH = ROOT / "models" / "evaluation_results.json"

ZEEK_RESULTS_PATH = ROOT / "data" / "zeek" / "detections.json"

API_URL = os.getenv(
    "SENTINELAI_API_URL",
    "http://127.0.0.1:8000"
)

# Load API credentials from .env
load_dotenv(ROOT / ".env")

SENTINELAI_API_KEY = os.getenv("SENTINELAI_API_KEY")

if not SENTINELAI_API_KEY:
    raise RuntimeError(
        "SENTINELAI_API_KEY is not configured."
    )

API_HEADERS = {
    "X-API-Key": SENTINELAI_API_KEY
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SentinelAI Security Dashboard",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def load_model():

    bundle = joblib.load(MODEL_PATH)

    return (
        bundle["model"],
        bundle["features"]
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD EVALUATION RESULTS
# ============================================================

@st.cache_data
def load_results():

    with open(RESULT_PATH, "r") as f:
        return json.load(f)


# ============================================================
# LOAD ZEEK RESULTS
# ============================================================

@st.cache_data
def load_zeek_results():

    if not ZEEK_RESULTS_PATH.exists():
        return []

    with open(ZEEK_RESULTS_PATH, "r") as f:
        return json.load(f)


# ============================================================
# LOAD EVERYTHING
# ============================================================

model, feature_names = load_model()

df = load_data()

results = load_results()

zeek_results = load_zeek_results()


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ SentinelAI")

st.subheader(
    "AI-Powered Network Threat Detection & Monitoring"
)

st.caption(
    "Machine-learning based DDoS traffic detection using CIC-IDS2017"
)

st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("## 📊 Random Forest Performance")

accuracy = results["accuracy"] * 100
precision = results["precision"] * 100
recall = results["recall"] * 100
f1 = results["f1_score"] * 100


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Accuracy",
        f"{accuracy:.2f}%"
    )

with col2:

    st.metric(
        "Precision",
        f"{precision:.2f}%"
    )

with col3:

    st.metric(
        "Recall",
        f"{recall:.2f}%"
    )

with col4:

    st.metric(
        "F1 Score",
        f"{f1:.2f}%"
    )


st.divider()


# ============================================================
# DATASET INFORMATION
# ============================================================

st.markdown("## 📁 Dataset")

d1, d2, d3 = st.columns(3)

with d1:

    st.metric(
        "Total Samples",
        f"{len(df):,}"
    )

with d2:

    benign_count = int(
        (df["Label"] == 0).sum()
    )

    st.metric(
        "BENIGN Samples",
        f"{benign_count:,}"
    )

with d3:

    ddos_count = int(
        (df["Label"] == 1).sum()
    )

    st.metric(
        "DDoS Samples",
        f"{ddos_count:,}"
    )


st.divider()


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.markdown("## 🔍 Random Forest Confusion Matrix")

cm = results["confusion_matrix"]

fig, ax = plt.subplots(
    figsize=(6, 4)
)

image = ax.imshow(cm)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    ["BENIGN", "DDoS"]
)

ax.set_yticklabels(
    ["BENIGN", "DDoS"]
)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "Actual Label"
)

ax.set_title(
    "DDoS Detection Confusion Matrix"
)

for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            cm[i][j],
            ha="center",
            va="center",
            fontsize=14,
        )

fig.colorbar(
    image,
    ax=ax
)

st.pyplot(
    fig,
    use_container_width=False
)


st.divider()


# ============================================================
# TRAFFIC SELECTION
# ============================================================

st.markdown(
    "## 🔎 Network Traffic Analysis"
)

label_choice = st.selectbox(
    "Select traffic type",
    ["BENIGN", "DDoS"],
)

label_value = (
    0
    if label_choice == "BENIGN"
    else 1
)

filtered_df = (
    df[
        df["Label"] == label_value
    ]
    .reset_index(drop=True)
)

record_number = st.number_input(
    "Select record number",
    min_value=0,
    max_value=len(filtered_df) - 1,
    value=0,
    step=1,
)

selected_row = filtered_df.iloc[
    record_number
]


# ============================================================
# SELECTED NETWORK FLOW
# ============================================================

st.markdown(
    "### Selected Network Flow"
)

display_df = (
    selected_row
    .drop("Label")
    .to_frame("Value")
)

st.dataframe(
    display_df,
    use_container_width=True
)


# ============================================================
# AI MODEL ANALYSIS
# ============================================================

if st.button(
    "🚨 Analyze Traffic",
    use_container_width=True
):

    try:

        # ----------------------------------------------------
        # Prepare features
        # ----------------------------------------------------

        features = {
            column: float(
                selected_row[column]
            )
            for column in feature_names
        }

        payload = {
            "features": features
        }


        # ====================================================
        # RANDOM FOREST
        # ====================================================

        rf_response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )


        # ====================================================
        # PYTORCH
        # ====================================================

        dl_response = requests.post(
            f"{API_URL}/predict/deep",
            json=payload,
            timeout=10
        )


        # ====================================================
        # CHECK RESPONSES
        # ====================================================

        if (
            rf_response.status_code == 200
            and dl_response.status_code == 200
        ):

            rf_result = rf_response.json()

            dl_result = dl_response.json()


            # =================================================
            # RESULTS
            # =================================================

            st.markdown(
                "## 🤖 AI Detection Results"
            )

            col1, col2 = st.columns(2)


            # =================================================
            # RANDOM FOREST
            # =================================================

            with col1:

                st.markdown(
                    "### 🌲 Random Forest"
                )

                if rf_result["prediction"] == 1:

                    st.error(
                        "🚨 DDoS DETECTED"
                    )

                else:

                    st.success(
                        "✅ BENIGN"
                    )

                st.metric(
                    "Prediction",
                    rf_result["label"]
                )


            # =================================================
            # PYTORCH
            # =================================================

            with col2:

                st.markdown(
                    "### 🧠 PyTorch MLP"
                )

                if dl_result["prediction"] == 1:

                    st.error(
                        "🚨 DDoS DETECTED"
                    )

                else:

                    st.success(
                        "✅ BENIGN"
                    )

                st.metric(
                    "Prediction",
                    dl_result["label"]
                )

                st.metric(
                    "DDoS Probability",
                    f"{dl_result['probability'] * 100:.2f}%"
                )


            st.divider()


            # =================================================
            # ACTUAL LABEL
            # =================================================

            actual_label = (
                "DDoS"
                if selected_row["Label"] == 1
                else "BENIGN"
            )

            st.markdown(
                "### 📌 Dataset Ground Truth"
            )

            st.info(
                f"Actual traffic label: **{actual_label}**"
            )


            # =================================================
            # MODEL AGREEMENT
            # =================================================

            st.markdown(
                "### 🔄 Model Agreement"
            )

            if (
                rf_result["prediction"]
                == dl_result["prediction"]
            ):

                if rf_result["prediction"] == 1:

                    st.error(
                        "🚨 Both AI models detected DDoS traffic."
                    )

                else:

                    st.success(
                        "✅ Both AI models classified the traffic as BENIGN."
                    )

            else:

                st.warning(
                    "⚠️ The AI models produced different predictions."
                )


            # =================================================
            # SAMPLE PREDICTION CHECK
            # =================================================

            st.markdown(
                "### 🎯 Prediction Check"
            )

            rf_correct = (
                rf_result["prediction"]
                == int(
                    selected_row["Label"]
                )
            )

            dl_correct = (
                dl_result["prediction"]
                == int(
                    selected_row["Label"]
                )
            )

            c1, c2 = st.columns(2)

            with c1:

                if rf_correct:

                    st.success(
                        "Random Forest: Correct"
                    )

                else:

                    st.warning(
                        "Random Forest: Incorrect"
                    )

            with c2:

                if dl_correct:

                    st.success(
                        "PyTorch MLP: Correct"
                    )

                else:

                    st.warning(
                        "PyTorch MLP: Incorrect"
                    )


            # =================================================
            # API RESPONSES
            # =================================================

            with st.expander(
                "🔧 View API Responses"
            ):

                st.markdown(
                    "**Random Forest API**"
                )

                st.json(
                    rf_result
                )

                st.markdown(
                    "**PyTorch MLP API**"
                )

                st.json(
                    dl_result
                )


        else:

            st.error(
                "One or more API requests failed."
            )

            st.write(
                "Random Forest status:",
                rf_response.status_code
            )

            st.write(
                "PyTorch status:",
                dl_response.status_code
            )


    except requests.exceptions.RequestException:

        st.error(
            "⚠️ Could not connect to SentinelAI API. "
            "Make sure FastAPI is running on port 8000."
        )


# ============================================================
# INCIDENT MANAGEMENT
# ============================================================

st.divider()

st.markdown(
    "## 🚨 Incident Management"
)

st.caption(
    "Live incident history stored by the SentinelAI FastAPI backend."
)


# ============================================================
# LOAD INCIDENTS FROM API
# ============================================================

try:

    incident_response = requests.get(
        f"{API_URL}/incidents",
        timeout=5
    )

    if incident_response.status_code == 200:

        incident_data = incident_response.json()

        incidents = incident_data.get(
            "incidents",
            []
        )

    else:

        incidents = []

        st.warning(
            "Could not load incidents from the API."
        )

except requests.exceptions.RequestException:

    incidents = []

    st.warning(
        "⚠️ FastAPI is not reachable. "
        "Start the backend on port 8000."
    )


# ============================================================
# INCIDENT COUNTERS
# ============================================================

total_incidents = len(incidents)

open_incidents = sum(
    str(item.get("status", "")).upper()
    == "OPEN"
    for item in incidents
)

investigating_incidents = sum(
    str(item.get("status", "")).upper()
    == "INVESTIGATING"
    for item in incidents
)

resolved_incidents = sum(
    str(item.get("status", "")).upper()
    == "RESOLVED"
    for item in incidents
)

high_risk_incidents = sum(
    str(item.get("risk_level", "")).upper()
    == "HIGH"
    for item in incidents
)


i1, i2, i3, i4, i5 = st.columns(5)

with i1:

    st.metric(
        "Total Incidents",
        total_incidents
    )

with i2:

    st.metric(
        "OPEN",
        open_incidents
    )

with i3:

    st.metric(
        "INVESTIGATING",
        investigating_incidents
    )

with i4:

    st.metric(
        "RESOLVED",
        resolved_incidents
    )

with i5:

    st.metric(
        "HIGH RISK",
        high_risk_incidents
    )


# ============================================================
# REFRESH
# ============================================================

if st.button(
    "🔄 Refresh Incidents"
):

    st.rerun()


# ============================================================
# INCIDENT TABLE
# ============================================================

if incidents:

    st.markdown(
        "### 📋 Incident History"
    )

    incident_rows = []

    for incident in incidents:

        incident_rows.append(
            {
                "ID": incident.get("id"),
                "Timestamp": incident.get("timestamp"),
                "Threat": incident.get("threat"),
                "Risk Score": incident.get("risk_score"),
                "Risk Level": incident.get("risk_level"),
                "Status": incident.get("status"),
                "Source": incident.get("source_ip"),
                "Destination": incident.get(
                    "destination_ip"
                ),
                "Protocol": incident.get("protocol"),
                "Detection Source": incident.get(
                    "detection_source"
                ),
            }
        )

    incidents_df = pd.DataFrame(
        incident_rows
    )

    st.dataframe(
        incidents_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # INCIDENT DETAILS
    # ========================================================

    st.markdown(
        "### 🔍 Incident Details"
    )

    incident_ids = [
        incident["id"]
        for incident in incidents
    ]

    selected_incident_id = st.selectbox(
        "Select incident",
        incident_ids
    )

    selected_incident = next(
        (
            item
            for item in incidents
            if item["id"] == selected_incident_id
        ),
        None
    )

    if selected_incident:

        d1, d2 = st.columns(2)

        with d1:

            st.markdown(
                "#### Network Information"
            )

            st.write(
                "**Source IP:**",
                selected_incident.get(
                    "source_ip"
                )
            )

            st.write(
                "**Destination IP:**",
                selected_incident.get(
                    "destination_ip"
                )
            )

            st.write(
                "**Source Port:**",
                selected_incident.get(
                    "source_port"
                )
            )

            st.write(
                "**Destination Port:**",
                selected_incident.get(
                    "destination_port"
                )
            )

            st.write(
                "**Protocol:**",
                selected_incident.get(
                    "protocol"
                )
            )


        with d2:

            st.markdown(
                "#### Threat Information"
            )

            st.write(
                "**Threat:**",
                selected_incident.get(
                    "threat"
                )
            )

            st.write(
                "**Risk Score:**",
                selected_incident.get(
                    "risk_score"
                )
            )

            st.write(
                "**Risk Level:**",
                selected_incident.get(
                    "risk_level"
                )
            )

            st.write(
                "**Detection Source:**",
                selected_incident.get(
                    "detection_source"
                )
            )

            st.write(
                "**Description:**",
                selected_incident.get(
                    "description"
                )
            )


        # ====================================================
        # STATUS UPDATE
        # ====================================================

        st.markdown(
            "#### 🔄 Update Incident Status"
        )

        current_status = str(
            selected_incident.get(
                "status",
                "OPEN"
            )
        ).upper()

        status_options = [
            "OPEN",
            "INVESTIGATING",
            "RESOLVED",
            "FALSE_POSITIVE"
        ]

        selected_status = st.selectbox(
            "New status",
            status_options,
            index=(
                status_options.index(
                    current_status
                )
                if current_status in status_options
                else 0
            )
        )

        if st.button(
            "Update Incident Status",
            key=f"update_{selected_incident_id}"
        ):

            try:

                update_response = requests.patch(
                    f"{API_URL}/incidents/"
                    f"{selected_incident_id}/status",
                    json={
                        "status": selected_status
                    },
                    timeout=5
                )

                if update_response.status_code == 200:

                    st.success(
                        "Incident status updated successfully."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Failed to update incident status."
                    )

                    st.json(
                        update_response.json()
                    )

            except requests.exceptions.RequestException:

                st.error(
                    "Could not connect to SentinelAI API."
                )


else:

    st.info(
        "No incidents have been recorded yet."
    )


# ============================================================
# ZEEK SECURITY EVENTS
# ============================================================

st.divider()

st.markdown(
    "## 🌐 Zeek Security Events"
)

if not zeek_results:

    st.info(
        "No Zeek security events available."
    )

else:

    total_flows = len(
        zeek_results
    )

    high_count = sum(
        item["risk_level"] == "HIGH"
        for item in zeek_results
    )

    medium_count = sum(
        item["risk_level"] == "MEDIUM"
        for item in zeek_results
    )

    low_count = sum(
        item["risk_level"] == "LOW"
        for item in zeek_results
    )


    # --------------------------------------------------------
    # EVENT COUNTERS
    # --------------------------------------------------------

    z1, z2, z3, z4 = st.columns(4)

    with z1:

        st.metric(
            "Total Flows",
            total_flows
        )

    with z2:

        st.metric(
            "HIGH",
            high_count
        )

    with z3:

        st.metric(
            "MEDIUM",
            medium_count
        )

    with z4:

        st.metric(
            "LOW",
            low_count
        )


    # --------------------------------------------------------
    # EVENTS TABLE
    # --------------------------------------------------------

    st.markdown(
        "### Network Events"
    )

    events_df = pd.DataFrame(
        zeek_results
    )

    st.dataframe(
        events_df,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SentinelAI • AI-powered network security monitoring"
)