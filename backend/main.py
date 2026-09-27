from pathlib import Path
import json
import sqlite3
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

SENTINELAI_API_KEY = __import__('os').getenv('SENTINELAI_API_KEY')

if not SENTINELAI_API_KEY:
    raise RuntimeError('SENTINELAI_API_KEY is not configured.')


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RF_MODEL_PATH = ROOT / "models" / "ddos_model.pkl"
DL_MODEL_PATH = ROOT / "models" / "deep_ddos_model.pt"
SCALER_PATH = ROOT / "models" / "deep_scaler.pkl"

DB_PATH = ROOT / "data" / "sentinelai.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SentinelAI API",
    version="1.0.0",
    description="AI-powered network threat detection and incident management API",
)


# ============================================================
# API KEY SECURITY
# ============================================================

def verify_api_key(x_api_key: str | None = Header(default=None)):
    if x_api_key != SENTINELAI_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )

    return True


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

if not RF_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Random Forest model not found: {RF_MODEL_PATH}"
    )

rf_bundle = joblib.load(RF_MODEL_PATH)

rf_model = rf_bundle["model"]
features = rf_bundle["features"]


# ============================================================
# DEEP LEARNING MODEL
# ============================================================

class SentinelMLP(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Deep learning scaler not found: {SCALER_PATH}"
    )

if not DL_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Deep learning model not found: {DL_MODEL_PATH}"
    )


dl_scaler = joblib.load(SCALER_PATH)

dl_checkpoint = torch.load(
    DL_MODEL_PATH,
    map_location="cpu",
)

dl_model = SentinelMLP(
    dl_checkpoint["input_size"]
)

dl_model.load_state_dict(
    dl_checkpoint["model_state_dict"]
)

dl_model.eval()


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():
    """
    Create a SQLite connection with Row support.
    """
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    """
    Create the incidents table if it does not exist.

    Also performs a lightweight migration for existing
    SentinelAI databases that were created before incident_json
    was added.
    """

    connection = get_db_connection()

    try:

        # ----------------------------------------------------
        # Create complete schema
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                timestamp TEXT NOT NULL,

                incident_json TEXT,

                source_ip TEXT,

                destination_ip TEXT,

                source_port INTEGER,

                destination_port INTEGER,

                protocol TEXT,

                threat TEXT NOT NULL,

                risk_score REAL NOT NULL,

                risk_level TEXT NOT NULL,

                status TEXT NOT NULL,

                detection_source TEXT,

                description TEXT,

                correlation_key TEXT,

                evidence_count INTEGER DEFAULT 1,

                last_seen TEXT
            )
            """
        )

        # ----------------------------------------------------
        # Check existing columns
        # ----------------------------------------------------

        columns = connection.execute(
            "PRAGMA table_info(incidents)"
        ).fetchall()

        column_names = {
            column["name"]
            for column in columns
        }

        # ----------------------------------------------------
        # Migration: incident_json
        # ----------------------------------------------------

        if "incident_json" not in column_names:

            connection.execute(
                """
                ALTER TABLE incidents
                ADD COLUMN incident_json TEXT
                """
            )

        # ----------------------------------------------------
        # Migration: structured network fields
        # ----------------------------------------------------

        if "source_ip" not in column_names:

            connection.execute(
                """
                ALTER TABLE incidents
                ADD COLUMN source_ip TEXT
                """
            )

        if "destination_ip" not in column_names:

            connection.execute(
                """
                ALTER TABLE incidents
                ADD COLUMN destination_ip TEXT
                """
            )

        if "source_port" not in column_names:

            connection.execute(
                """
                ALTER TABLE incidents
                ADD COLUMN source_port INTEGER
                """
            )

        if "destination_port" not in column_names:

            connection.execute(
                """
                ALTER TABLE incidents
                ADD COLUMN destination_port INTEGER
                """
            )

        if "protocol" not in column_names:

            connection.execute(
                """
                ALTER TABLE incidents
                ADD COLUMN protocol TEXT
                """
            )

        if "detection_source" not in column_names:

            connection.execute(
                """
                ALTER TABLE incidents
                ADD COLUMN detection_source TEXT
                """
            )

        if "description" not in column_names:

            connection.execute(
                """
                ALTER TABLE incidents
                ADD COLUMN description TEXT
                """
            )

        # ----------------------------------------------------
        # Migration: correlation_key
        # ----------------------------------------------------

        if "correlation_key" not in column_names:

            connection.execute(
                """
                ALTER TABLE incidents
                ADD COLUMN correlation_key TEXT
                """
            )

        # ----------------------------------------------------
        # Migration: evidence_count
        # ----------------------------------------------------

        if "evidence_count" not in column_names:

            connection.execute(
                """
                ALTER TABLE incidents
                ADD COLUMN evidence_count INTEGER DEFAULT 1
                """
            )

        # ----------------------------------------------------
        # Migration: last_seen
        # ----------------------------------------------------

        if "last_seen" not in column_names:

            connection.execute(
                """
                ALTER TABLE incidents
                ADD COLUMN last_seen TEXT
                """
            )

        connection.commit()

    finally:

        connection.close()


# Initialize database when API starts/imports
init_database()


# ============================================================
# REQUEST MODELS
# ============================================================

class TrafficData(BaseModel):

    features: dict[str, float]


class IncidentCreate(BaseModel):

    threat: str

    risk_score: float

    risk_level: str

    status: str = "OPEN"

    detection_source: str

    description: str

    source_ip: str | None = None

    destination_ip: str | None = None

    source_port: float | None = None

    destination_port: float | None = None

    protocol: str | None = None


class IncidentStatusUpdate(BaseModel):

    status: str


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "project": "SentinelAI",
        "status": "running",
        "models": [
            "Random Forest",
            "PyTorch MLP",
        ],
        "features": [
            "AI Threat Detection",
            "Incident Management",
            "Zeek Network Monitoring",
            "SQLite Persistence",
        ],
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "models_loaded": True,
        "database": DB_PATH.exists(),
    }


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: TrafficData, x_api_key_verified: bool = Depends(verify_api_key)):

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing = [
        name
        for name in features
        if name not in data.features
    ]

    if missing:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required features",
                "missing_features": missing,
            },
        )

    # --------------------------------------------------------
    # Prepare sample
    # --------------------------------------------------------

    try:

        sample = pd.DataFrame(
            [
                [
                    float(data.features[name])
                    for name in features
                ]
            ],
            columns=features,
        )

    except (TypeError, ValueError) as error:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid feature values",
                "message": str(error),
            },
        )

    # --------------------------------------------------------
    # Random Forest prediction
    # --------------------------------------------------------

    try:

        prediction = int(
            rf_model.predict(sample)[0]
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Random Forest prediction failed",
                "message": str(error),
            },
        )

    label = (
        "DDoS"
        if prediction == 1
        else "BENIGN"
    )

    # --------------------------------------------------------
    # Incident information
    # --------------------------------------------------------

    incident_id = None
    incident_created = False
    duplicate_incident = False

    # --------------------------------------------------------
    # Automatically create DDoS incident
    # --------------------------------------------------------

    if prediction == 1:

        connection = get_db_connection()

        try:

            # ------------------------------------------------
            # Check existing OPEN DDoS incident
            # ------------------------------------------------

            existing_incident = connection.execute(
                """
                SELECT id
                FROM incidents
                WHERE threat = ?
                  AND risk_level = ?
                  AND status = ?
                  AND detection_source = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (
                    "DDoS",
                    "HIGH",
                    "OPEN",
                    "Random Forest",
                ),
            ).fetchone()

            # ------------------------------------------------
            # Existing incident
            # ------------------------------------------------

            if existing_incident is not None:

                incident_id = existing_incident["id"]

                duplicate_incident = True

            # ------------------------------------------------
            # Create new incident
            # ------------------------------------------------

            else:

                timestamp = datetime.now(
                    timezone.utc
                ).isoformat()

                incident_data = {
                    "threat": "DDoS",
                    "risk_score": 85,
                    "risk_level": "HIGH",
                    "status": "OPEN",

                    "source_ip": data.features.get(
                        "Source IP"
                    ),

                    "destination_ip": data.features.get(
                        "Destination IP"
                    ),

                    "source_port": data.features.get(
                        "Source Port"
                    ),

                    "destination_port": data.features.get(
                        "Destination Port"
                    ),

                    "protocol": data.features.get(
                        "Protocol"
                    ),

                    "detection_source": "Random Forest",

                    "description": (
                        "Potential DDoS traffic detected "
                        "by the Random Forest model."
                    ),
                }

                incident_json = json.dumps(
                    incident_data
                )

                cursor = connection.execute(
                    """
                    INSERT INTO incidents (

                        timestamp,

                        incident_json,

                        source_ip,
                        destination_ip,

                        source_port,
                        destination_port,

                        protocol,

                        threat,
                        risk_score,
                        risk_level,
                        status,

                        detection_source,
                        description

                    )

                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        timestamp,

                        incident_json,

                        incident_data["source_ip"],
                        incident_data["destination_ip"],

                        incident_data["source_port"],
                        incident_data["destination_port"],

                        incident_data["protocol"],

                        incident_data["threat"],
                        incident_data["risk_score"],
                        incident_data["risk_level"],
                        incident_data["status"],

                        incident_data["detection_source"],
                        incident_data["description"],
                    ),
                )

                connection.commit()

                incident_id = cursor.lastrowid

                incident_created = True

        finally:

            connection.close()

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    response = {
        "model": "Random Forest",

        "prediction": prediction,

        "label": label,

        "message": (
            "Potential DDoS traffic detected"
            if prediction == 1
            else "Traffic classified as benign"
        ),

        "incident_created": incident_created,

        "duplicate_incident": duplicate_incident,
    }

    if incident_id is not None:

        response["incident_id"] = incident_id

    return response


# ============================================================
# DEEP LEARNING PREDICTION
# ============================================================

@app.post("/predict/deep")
def predict_deep(data: TrafficData, x_api_key_verified: bool = Depends(verify_api_key)):

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing = [
        name
        for name in features
        if name not in data.features
    ]

    if missing:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required features",
                "missing_features": missing,
            },
        )

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    try:

        sample = np.array(
            [
                [
                    float(data.features[name])
                    for name in features
                ]
            ],
            dtype=np.float32,
        )

    except (TypeError, ValueError) as error:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid feature values",
                "message": str(error),
            },
        )

    # --------------------------------------------------------
    # Scale input
    # --------------------------------------------------------

    try:

        sample_scaled = dl_scaler.transform(
            sample
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Deep learning scaling failed",
                "message": str(error),
            },
        )

    tensor = torch.tensor(
        sample_scaled,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Deep learning prediction
    # --------------------------------------------------------

    try:

        with torch.no_grad():

            output = dl_model(tensor)

            probability = torch.sigmoid(
                output
            ).item()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Deep learning prediction failed",
                "message": str(error),
            },
        )

    prediction = (
        1
        if probability >= 0.5
        else 0
    )

    label = (
        "DDoS"
        if prediction == 1
        else "BENIGN"
    )

    return {
        "model": "PyTorch MLP",

        "prediction": prediction,

        "label": label,

        "probability": round(
            probability,
            4,
        ),

        "message": (
            "Potential DDoS traffic detected"
            if prediction == 1
            else "Traffic classified as benign"
        ),
    }


# ============================================================
# CREATE INCIDENT
# ============================================================

@app.post("/incidents")
def create_incident(data: IncidentCreate, x_api_key_verified: bool = Depends(verify_api_key)):

    # --------------------------------------------------------
    # Normalize values
    # --------------------------------------------------------

    risk_level = data.risk_level.upper()
    status = data.status.upper()

    allowed_risk_levels = {
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    allowed_statuses = {
        "OPEN",
        "INVESTIGATING",
        "RESOLVED",
        "FALSE_POSITIVE",
    }

    if risk_level not in allowed_risk_levels:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid risk level",
                "allowed_risk_levels": sorted(
                    allowed_risk_levels
                ),
            },
        )

    if status not in allowed_statuses:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid status",
                "allowed_statuses": sorted(
                    allowed_statuses
                ),
            },
        )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    # --------------------------------------------------------
    # Normalize network values
    # --------------------------------------------------------

    source_ip = (
        str(data.source_ip).strip()
        if data.source_ip
        else None
    )

    destination_ip = (
        str(data.destination_ip).strip()
        if data.destination_ip
        else None
    )

    protocol = (
        str(data.protocol).strip().lower()
        if data.protocol
        else None
    )

    source_port = (
        int(data.source_port)
        if data.source_port is not None
        else None
    )

    destination_port = (
        int(data.destination_port)
        if data.destination_port is not None
        else None
    )

    threat = str(
        data.threat
    ).strip()

    detection_source = str(
        data.detection_source
    ).strip()

    # --------------------------------------------------------
    # Build correlation key
    # --------------------------------------------------------
    #
    # Source port is intentionally excluded.
    # Repeated connections to the same destination service
    # and same threat type are therefore correlated.
    #
    # Example:
    # 192.168.170.56|217.13.4.24|53|udp|High Packet Rate
    # --------------------------------------------------------

    correlation_key = (
        f"{source_ip or 'unknown'}|"
        f"{destination_ip or 'unknown'}|"
        f"{destination_port or 'unknown'}|"
        f"{protocol or 'unknown'}|"
        f"{threat}"
    )

    # --------------------------------------------------------
    # Incident data
    # --------------------------------------------------------

    incident_data = {
        "threat": threat,
        "risk_score": data.risk_score,
        "risk_level": risk_level,
        "status": status,
        "source_ip": source_ip,
        "destination_ip": destination_ip,
        "source_port": source_port,
        "destination_port": destination_port,
        "protocol": protocol,
        "detection_source": detection_source,
        "description": data.description,
        "correlation_key": correlation_key,
    }

    incident_json = json.dumps(
        incident_data
    )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    connection = get_db_connection()

    try:

        # ----------------------------------------------------
        # Check for an existing OPEN incident
        # ----------------------------------------------------

        existing = connection.execute(
            """
            SELECT *
            FROM incidents
            WHERE correlation_key = ?
              AND status = 'OPEN'
            ORDER BY id DESC
            LIMIT 1
            """,
            (correlation_key,),
        ).fetchone()

        # ----------------------------------------------------
        # Existing incident found
        # ----------------------------------------------------

        if existing is not None:

            existing_id = existing["id"]

            old_count = existing["evidence_count"]

            if old_count is None:
                old_count = 1

            new_count = int(old_count) + 1

            old_score = (
                existing["risk_score"]
                if existing["risk_score"] is not None
                else 0
            )

            new_score = max(
                float(old_score),
                float(data.risk_score),
            )

            risk_priority = {
                "LOW": 1,
                "MEDIUM": 2,
                "HIGH": 3,
            }

            old_level = str(
                existing["risk_level"] or "LOW"
            ).upper()

            if risk_priority.get(
                risk_level, 1
            ) > risk_priority.get(
                old_level, 1
            ):
                new_level = risk_level
            else:
                new_level = old_level

            # Update the existing incident with latest evidence.
            connection.execute(
                """
                UPDATE incidents
                SET
                    timestamp = ?,
                    incident_json = ?,
                    risk_score = ?,
                    risk_level = ?,
                    source_ip = ?,
                    destination_ip = ?,
                    source_port = ?,
                    destination_port = ?,
                    protocol = ?,
                    threat = ?,
                    detection_source = ?,
                    description = ?,
                    evidence_count = ?,
                    last_seen = ?
                WHERE id = ?
                """,
                (
                    timestamp,
                    incident_json,
                    new_score,
                    new_level,
                    source_ip,
                    destination_ip,
                    source_port,
                    destination_port,
                    protocol,
                    threat,
                    detection_source,
                    data.description,
                    new_count,
                    timestamp,
                    existing_id,
                ),
            )

            connection.commit()

            return {
                "message": "Existing incident updated",
                "incident_id": existing_id,
                "status": "OPEN",
                "deduplicated": True,
                "evidence_count": new_count,
                "correlation_key": correlation_key,
            }

        # ----------------------------------------------------
        # No existing incident -> create new incident
        # ----------------------------------------------------

        cursor = connection.execute(
            """
            INSERT INTO incidents (
                timestamp,
                incident_json,
                source_ip,
                destination_ip,
                source_port,
                destination_port,
                protocol,
                threat,
                risk_score,
                risk_level,
                status,
                detection_source,
                description,
                correlation_key,
                evidence_count,
                last_seen
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                timestamp,
                incident_json,
                source_ip,
                destination_ip,
                source_port,
                destination_port,
                protocol,
                threat,
                data.risk_score,
                risk_level,
                status,
                detection_source,
                data.description,
                correlation_key,
                1,
                timestamp,
            ),
        )

        connection.commit()

        incident_id = cursor.lastrowid

    finally:

        connection.close()

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "message": "Incident created successfully",
        "incident_id": incident_id,
        "status": status,
        "deduplicated": False,
        "evidence_count": 1,
        "correlation_key": correlation_key,
    }


# ============================================================
# GET ALL INCIDENTS
# ============================================================

@app.get("/incidents")
def get_incidents():

    connection = get_db_connection()

    try:

        rows = connection.execute(
            """
            SELECT *
            FROM incidents
            ORDER BY id DESC
            """
        ).fetchall()

    finally:

        connection.close()

    return {
        "count": len(rows),

        "incidents": [
            dict(row)
            for row in rows
        ],
    }


# ============================================================
# GET SINGLE INCIDENT
# ============================================================

@app.get("/incidents/{incident_id}")
def get_incident(incident_id: int):

    connection = get_db_connection()

    try:

        row = connection.execute(
            """
            SELECT *
            FROM incidents
            WHERE id = ?
            """,
            (incident_id,),
        ).fetchone()

    finally:

        connection.close()

    if row is None:

        raise HTTPException(
            status_code=404,
            detail="Incident not found",
        )

    return dict(row)


# ============================================================
# UPDATE INCIDENT STATUS
# ============================================================

@app.patch("/incidents/{incident_id}/status")
def update_incident_status(
    incident_id: int,
    data: IncidentStatusUpdate,
):

    allowed_statuses = {
        "OPEN",
        "INVESTIGATING",
        "RESOLVED",
        "FALSE_POSITIVE",
    }

    status = data.status.upper()

    if status not in allowed_statuses:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid status",
                "allowed_statuses": sorted(
                    allowed_statuses
                ),
            },
        )

    connection = get_db_connection()

    try:

        cursor = connection.execute(
            """
            UPDATE incidents
            SET status = ?
            WHERE id = ?
            """,
            (
                status,
                incident_id,
            ),
        )

        connection.commit()

        updated = cursor.rowcount

    finally:

        connection.close()

    if updated == 0:

        raise HTTPException(
            status_code=404,
            detail="Incident not found",
        )

    return {
        "message": "Incident status updated",

        "incident_id": incident_id,

        "status": status,
    }