import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
load_dotenv()

SENTINELAI_API_KEY = os.getenv("SENTINELAI_API_KEY")

if not SENTINELAI_API_KEY:
    raise RuntimeError(
        "SENTINELAI_API_KEY is not configured."
    )

# ============================================================
# CONFIGURATION
# ============================================================

DETECTIONS_FILE = Path("data/zeek/detections.json")

API_URL = "http://127.0.0.1:8000/incidents"

REQUEST_TIMEOUT = 10


# ============================================================
# HELPERS
# ============================================================

def normalize_protocol(protocol):
    """Return protocol in a consistent format."""

    if protocol is None:
        return "unknown"

    return str(protocol).strip().lower()


def build_incident_key(detection):
    """
    Build a correlation key for a Zeek detection.

    Source port is intentionally excluded so that repeated
    connections from the same source to the same destination
    service can be correlated.
    """

    source_ip = detection.get(
        "source_ip",
        "unknown",
    )

    destination_ip = detection.get(
        "destination_ip",
        "unknown",
    )

    destination_port = detection.get(
        "destination_port",
        "unknown",
    )

    protocol = normalize_protocol(
        detection.get("protocol")
    )

    threat_type = detection.get(
        "threat_type",
        "Suspicious Network Traffic",
    )

    return (
        f"{source_ip}|"
        f"{destination_ip}|"
        f"{destination_port}|"
        f"{protocol}|"
        f"{threat_type}"
    )


def build_description(
    source_ip,
    destination_ip,
    destination_port,
    threat_type,
    reasons,
):
    """Build a readable incident description."""

    reason_text = ", ".join(
        map(str, reasons)
    )

    return (
        f"Zeek detected {threat_type.lower()} "
        f"from {source_ip or 'unknown'} "
        f"to {destination_ip or 'unknown'} "
        f"on port {destination_port or 'unknown'}. "
        f"Reasons: {reason_text or 'No additional details'}."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check detection file
    # --------------------------------------------------------

    if not DETECTIONS_FILE.exists():

        print(
            f"❌ Detection file not found: "
            f"{DETECTIONS_FILE}"
        )

        return

    # --------------------------------------------------------
    # Load detections
    # --------------------------------------------------------

    try:

        with open(
            DETECTIONS_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            detections = json.load(file)

    except json.JSONDecodeError as error:

        print(
            f"❌ Invalid JSON in detection file: "
            f"{error}"
        )

        return

    except OSError as error:

        print(
            f"❌ Could not read detection file: "
            f"{error}"
        )

        return

    # --------------------------------------------------------
    # Validate detection list
    # --------------------------------------------------------

    if not isinstance(detections, list):

        print(
            "❌ Detection file must contain "
            "a JSON list."
        )

        return

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    print(
        "\n🚀 Sending Zeek detections "
        "to SentinelAI"
    )

    print(
        "----------------------------------------"
    )

    created = 0
    skipped = 0
    duplicates = 0
    errors = 0

    # Tracks duplicate detections during
    # this execution.
    seen_keys = set()

    # --------------------------------------------------------
    # Process detections
    # --------------------------------------------------------

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        # ----------------------------------------------------
        # Validate detection
        # ----------------------------------------------------

        if not isinstance(
            detection,
            dict,
        ):

            print(
                f"⚠️ Detection #{index} "
                f"is invalid. Skipping."
            )

            errors += 1

            continue

        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        risk_level = str(
            detection.get(
                "risk_level",
                "LOW",
            )
        ).upper()

        # ----------------------------------------------------
        # Skip LOW events
        # ----------------------------------------------------

        if risk_level == "LOW":

            skipped += 1

            continue

        # ----------------------------------------------------
        # Extract detection information
        # ----------------------------------------------------

        source_ip = detection.get(
            "source_ip"
        )

        destination_ip = detection.get(
            "destination_ip"
        )

        source_port = detection.get(
            "source_port"
        )

        destination_port = detection.get(
            "destination_port"
        )

        protocol = detection.get(
            "protocol"
        )

        threat_type = detection.get(
            "threat_type",
            "Suspicious Network Traffic",
        )

        risk_score = detection.get(
            "risk_score",
            0,
        )

        reasons = detection.get(
            "reasons",
            [],
        )

        indicators = detection.get(
            "indicators",
            [],
        )

        # ----------------------------------------------------
        # Normalize reasons
        # ----------------------------------------------------

        if not isinstance(
            reasons,
            list,
        ):

            reasons = [
                str(reasons)
            ]

        # ----------------------------------------------------
        # Normalize indicators
        # ----------------------------------------------------

        if not isinstance(
            indicators,
            list,
        ):

            indicators = [
                str(indicators)
            ]

        # ----------------------------------------------------
        # Build correlation key
        # ----------------------------------------------------

        incident_key = build_incident_key(
            detection
        )

        # ----------------------------------------------------
        # Detect duplicate in current batch
        # ----------------------------------------------------

        if incident_key in seen_keys:

            print(
                f"\n🔁 Detection #{index}"
            )

            print(
                "   Duplicate correlated flow"
            )

            print(
                f"   Threat      : {threat_type}"
            )

            print(
                f"   Source      : {source_ip}"
            )

            print(
                f"   Destination : {destination_ip}"
            )

            print(
                f"   Port        : {destination_port}"
            )

            print(
                "   ⏭️ Skipped"
            )

            duplicates += 1

            continue

        seen_keys.add(
            incident_key
        )

        # ----------------------------------------------------
        # Build description
        # ----------------------------------------------------

        description = build_description(
            source_ip=source_ip,
            destination_ip=destination_ip,
            destination_port=destination_port,
            threat_type=threat_type,
            reasons=reasons,
        )

        # ----------------------------------------------------
        # API payload
        # ----------------------------------------------------

        payload = {
            "threat": threat_type,

            "risk_score": risk_score,

            "risk_level": risk_level,

            "status": "OPEN",

            "detection_source": "Zeek",

            "source_ip": source_ip,

            "destination_ip": destination_ip,

            "source_port": source_port,

            "destination_port": destination_port,

            "protocol": protocol,

            "description": description,

            # Additional detection metadata.
            "indicators": indicators,
        }

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"\n📡 Detection #{index}"
        )

        print(
            f"   Threat      : {threat_type}"
        )

        print(
            f"   Source      : {source_ip}"
        )

        print(
            f"   Destination : {destination_ip}"
        )

        print(
            f"   Protocol    : {protocol}"
        )

        print(
            f"   Port        : {destination_port}"
        )

        print(
            f"   Risk        : "
            f"{risk_level} ({risk_score})"
        )

        # ----------------------------------------------------
        # Send to FastAPI
        # ----------------------------------------------------

        try:

            response = requests.post(
    API_URL,
    json=payload,
    headers={
        "X-API-Key": SENTINELAI_API_KEY,
    },
    timeout=REQUEST_TIMEOUT,
)

            # ------------------------------------------------
            # Successful response
            # ------------------------------------------------

            if response.status_code in (
                200,
                201,
            ):

                try:

                    response_data = (
                        response.json()
                    )

                except ValueError:

                    response_data = {}

                incident_id = response_data.get(
                    "incident_id",
                    "N/A",
                )

                print(
                    "   ✅ Incident created"
                )

                print(
                    f"   Incident ID: "
                    f"{incident_id}"
                )

                created += 1

            # ------------------------------------------------
            # Validation error
            # ------------------------------------------------

            elif response.status_code == 422:

                print(
                    "   ❌ Validation error (422)"
                )

                print(
                    f"   Response: "
                    f"{response.text}"
                )

                errors += 1

            # ------------------------------------------------
            # Other API error
            # ------------------------------------------------

            else:

                print(
                    f"   ❌ API error "
                    f"{response.status_code}"
                )

                print(
                    f"   Response: "
                    f"{response.text}"
                )

                errors += 1

        # ----------------------------------------------------
        # Connection error
        # ----------------------------------------------------

        except requests.exceptions.ConnectionError:

            print(
                "   ❌ Connection error: "
                "FastAPI is not running "
                "on port 8000."
            )

            errors += 1

        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        except requests.exceptions.Timeout:

            print(
                "   ❌ Request timed out "
                "while contacting FastAPI."
            )

            errors += 1

        # ----------------------------------------------------
        # General request error
        # ----------------------------------------------------

        except requests.exceptions.RequestException as error:

            print(
                f"   ❌ Request error: "
                f"{error}"
            )

            errors += 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n📊 Summary"
    )

    print(
        "----------"
    )

    print(
        f"Total detections   : "
        f"{len(detections)}"
    )

    print(
        f"Incidents created  : "
        f"{created}"
    )

    print(
        f"LOW events skipped : "
        f"{skipped}"
    )

    print(
        f"Duplicates skipped : "
        f"{duplicates}"
    )

    print(
        f"Errors             : "
        f"{errors}"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if errors == 0:

        print(
            "\n✅ Zeek → SentinelAI "
            "integration completed successfully."
        )

    elif created > 0:

        print(
            "\n⚠️ Zeek integration "
            "completed with some errors."
        )

    else:

        print(
            "\n❌ No incidents were created. "
            "Check that FastAPI is running."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()