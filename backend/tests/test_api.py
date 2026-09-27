import os

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

API_KEY = os.getenv("SENTINELAI_API_KEY")


def auth_headers():
    return {"X-API-Key": API_KEY}


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


def test_protected_predict_without_api_key():
    response = client.post(
        "/predict",
        json={"features": {}},
    )

    assert response.status_code == 401


def test_incidents_endpoint():
    response = client.get("/incidents")

    assert response.status_code == 200

    data = response.json()

    assert "incidents" in data
    assert "count" in data


def test_create_incident():
    response = client.post(
        "/incidents",
        headers=auth_headers(),
        json={
            "source": "pytest",
            "threat": "Test Security Event",
            "risk_score": 0.10,
            "risk_level": "LOW",
            "detection_source": "pytest",
            "description": "Automated API security test incident",
            "status": "OPEN",
            "source_ip": "10.255.255.1",
            "destination_ip": "10.255.255.2",
            "destination_port": 9999,
            "protocol": "TCP",
        },
    )

    assert response.status_code in (200, 201)

    data = response.json()

    assert "incident_id" in data
    assert data["status"] == "OPEN"


def test_incident_status_update():
    create_response = client.post(
        "/incidents",
        headers=auth_headers(),
        json={
            "source": "pytest-status",
            "threat": "Status Test Event",
            "risk_score": 0.10,
            "risk_level": "LOW",
            "detection_source": "pytest-status",
            "description": "Automated incident status test",
            "status": "OPEN",
            "source_ip": "10.255.255.3",
            "destination_ip": "10.255.255.4",
            "destination_port": 9998,
            "protocol": "TCP",
        },
    )

    assert create_response.status_code in (200, 201)

    incident_id = create_response.json()["incident_id"]

    response = client.patch(
        f"/incidents/{incident_id}/status",
        headers=auth_headers(),
        json={"status": "RESOLVED"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "RESOLVED"
