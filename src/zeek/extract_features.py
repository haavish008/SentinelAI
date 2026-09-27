from pathlib import Path
import json


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "zeek" / "flows.json"
OUTPUT_FILE = ROOT / "data" / "zeek" / "zeek_features.json"


# ============================================================
# VALUE CONVERSION
# ============================================================

def to_float(value, default=0.0):

    if value in [None, "", "-"]:
        return default

    try:
        return float(value)

    except (ValueError, TypeError):
        return default


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def calculate_features(flow):

    # --------------------------------------------------------
    # Basic values
    # --------------------------------------------------------

    duration = to_float(
        flow.get("duration_seconds")
    )

    orig_packets = to_float(
        flow.get("orig_packets")
    )

    resp_packets = to_float(
        flow.get("resp_packets")
    )

    orig_bytes = to_float(
        flow.get("orig_bytes")
    )

    resp_bytes = to_float(
        flow.get("resp_bytes")
    )

    orig_ip_bytes = to_float(
        flow.get("orig_ip_bytes")
    )

    resp_ip_bytes = to_float(
        flow.get("resp_ip_bytes")
    )

    source_port = to_float(
        flow.get("source_port")
    )

    destination_port = to_float(
        flow.get("destination_port")
    )


    # --------------------------------------------------------
    # Total traffic
    # --------------------------------------------------------

    total_packets = (
        orig_packets +
        resp_packets
    )

    total_bytes = (
        orig_bytes +
        resp_bytes
    )

    total_ip_bytes = (
        orig_ip_bytes +
        resp_ip_bytes
    )


    # --------------------------------------------------------
    # Packet rates
    # --------------------------------------------------------

    if duration > 0:

        total_packets_per_second = (
            total_packets / duration
        )

        orig_packets_per_second = (
            orig_packets / duration
        )

        resp_packets_per_second = (
            resp_packets / duration
        )

        total_bytes_per_second = (
            total_bytes / duration
        )

        orig_bytes_per_second = (
            orig_bytes / duration
        )

        resp_bytes_per_second = (
            resp_bytes / duration
        )

    else:

        total_packets_per_second = 0.0
        orig_packets_per_second = 0.0
        resp_packets_per_second = 0.0

        total_bytes_per_second = 0.0
        orig_bytes_per_second = 0.0
        resp_bytes_per_second = 0.0


    # --------------------------------------------------------
    # Average packet sizes
    # --------------------------------------------------------

    if total_packets > 0:

        average_packet_size = (
            total_bytes / total_packets
        )

    else:

        average_packet_size = 0.0


    if orig_packets > 0:

        average_orig_packet_size = (
            orig_bytes / orig_packets
        )

    else:

        average_orig_packet_size = 0.0


    if resp_packets > 0:

        average_resp_packet_size = (
            resp_bytes / resp_packets
        )

    else:

        average_resp_packet_size = 0.0


    # --------------------------------------------------------
    # Traffic ratios
    # --------------------------------------------------------

    if resp_packets > 0:

        packet_ratio = (
            orig_packets / resp_packets
        )

    else:

        packet_ratio = orig_packets


    if resp_bytes > 0:

        byte_ratio = (
            orig_bytes / resp_bytes
        )

    else:

        byte_ratio = orig_bytes


    # --------------------------------------------------------
    # Extract connection information
    # --------------------------------------------------------

    source_ip = flow.get(
        "source_ip"
    )

    destination_ip = flow.get(
        "destination_ip"
    )

    protocol = flow.get(
        "protocol"
    )

    conn_state = flow.get(
        "conn_state"
    )

    history = flow.get(
        "history"
    )

    ip_protocol = to_float(
        flow.get("ip_protocol")
    )


    # ========================================================
    # RETURN ENGINEERED FEATURES
    # ========================================================

    return {

        # Network identity
        "source_ip":
            source_ip,

        "destination_ip":
            destination_ip,

        "source_port":
            source_port,

        "destination_port":
            destination_port,

        "protocol":
            protocol,


        # Timing
        "duration_seconds":
            duration,


        # Packet statistics
        "orig_packets":
            orig_packets,

        "resp_packets":
            resp_packets,

        "total_packets":
            total_packets,


        # Byte statistics
        "orig_bytes":
            orig_bytes,

        "resp_bytes":
            resp_bytes,

        "total_bytes":
            total_bytes,


        # IP byte statistics
        "orig_ip_bytes":
            orig_ip_bytes,

        "resp_ip_bytes":
            resp_ip_bytes,

        "total_ip_bytes":
            total_ip_bytes,


        # Packet rates
        "total_packets_per_second":
            total_packets_per_second,

        "orig_packets_per_second":
            orig_packets_per_second,

        "resp_packets_per_second":
            resp_packets_per_second,


        # Byte rates
        "total_bytes_per_second":
            total_bytes_per_second,

        "orig_bytes_per_second":
            orig_bytes_per_second,

        "resp_bytes_per_second":
            resp_bytes_per_second,


        # Average packet sizes
        "average_packet_size":
            average_packet_size,

        "average_orig_packet_size":
            average_orig_packet_size,

        "average_resp_packet_size":
            average_resp_packet_size,


        # Ratios
        "packet_ratio":
            packet_ratio,

        "byte_ratio":
            byte_ratio,


        # Connection metadata
        "conn_state":
            conn_state,

        "history":
            history,

        "ip_protocol":
            ip_protocol,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        print("❌ flows.json not found")
        print(f"Expected: {INPUT_FILE}")

        return


    # --------------------------------------------------------
    # Load flows
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r"
    ) as file:

        flows = json.load(file)


    # --------------------------------------------------------
    # Generate features
    # --------------------------------------------------------

    extracted = []

    for flow in flows:

        features = calculate_features(
            flow
        )

        extracted.append(
            features
        )


    # --------------------------------------------------------
    # Save features
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w"
    ) as file:

        json.dump(
            extracted,
            file,
            indent=4
        )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "✅ Zeek feature extraction complete"
    )

    print(
        "Input flows:",
        len(flows)
    )

    print(
        "Features generated:",
        len(extracted)
    )

    print(
        "Output:",
        OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()