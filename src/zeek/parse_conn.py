from pathlib import Path
import json


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CONN_LOG = ROOT / "data" / "zeek" / "conn.log"
OUTPUT_FILE = ROOT / "data" / "zeek" / "flows.json"


# ============================================================
# PARSE ZEEK CONN.LOG
# ============================================================

def parse_conn_log():

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not CONN_LOG.exists():

        print("❌ conn.log not found")
        print(f"Expected: {CONN_LOG}")

        return

    fields = None
    flows = []


    # --------------------------------------------------------
    # Read conn.log
    # --------------------------------------------------------

    with open(
        CONN_LOG,
        "r",
        errors="ignore"
    ) as file:

        for line in file:

            line = line.strip()


            # ------------------------------------------------
            # Skip empty lines
            # ------------------------------------------------

            if not line:
                continue


            # ------------------------------------------------
            # Read Zeek field definitions
            # ------------------------------------------------

            if line.startswith("#fields"):

                fields = line.split("\t")[1:]

                continue


            # ------------------------------------------------
            # Skip other Zeek metadata
            # ------------------------------------------------

            if line.startswith("#"):

                continue


            # ------------------------------------------------
            # Make sure fields are available
            # ------------------------------------------------

            if fields is None:

                continue


            # ------------------------------------------------
            # Split data row
            # ------------------------------------------------

            values = line.split("\t")


            # ------------------------------------------------
            # Validate number of fields
            # ------------------------------------------------

            if len(values) != len(fields):

                continue


            # ------------------------------------------------
            # Create record
            # ------------------------------------------------

            record = dict(
                zip(
                    fields,
                    values
                )
            )


            # ------------------------------------------------
            # Extract network flow information
            # ------------------------------------------------

            flow = {

                # Network identity
                "source_ip":
                    record.get("id.orig_h"),

                "destination_ip":
                    record.get("id.resp_h"),

                "source_port":
                    record.get("id.orig_p"),

                "destination_port":
                    record.get("id.resp_p"),

                "protocol":
                    record.get("proto"),


                # Timing
                "duration_seconds":
                    record.get("duration"),


                # Packet statistics
                "orig_packets":
                    record.get("orig_pkts"),

                "resp_packets":
                    record.get("resp_pkts"),


                # Byte statistics
                "orig_bytes":
                    record.get("orig_bytes"),

                "resp_bytes":
                    record.get("resp_bytes"),


                # IP byte statistics
                "orig_ip_bytes":
                    record.get("orig_ip_bytes"),

                "resp_ip_bytes":
                    record.get("resp_ip_bytes"),


                # Connection information
                "conn_state":
                    record.get("conn_state"),

                "history":
                    record.get("history"),


                # IP protocol number
                "ip_protocol":
                    record.get("ip_proto"),
            }


            # ------------------------------------------------
            # Store flow
            # ------------------------------------------------

            flows.append(flow)


    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w"
    ) as file:

        json.dump(
            flows,
            file,
            indent=4
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("✅ Zeek log parsed successfully")

    print(
        "Flows extracted:",
        len(flows)
    )

    print(
        "Output:",
        OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parse_conn_log()