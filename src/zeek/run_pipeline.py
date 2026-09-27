import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_step(name, command):
    print("\n" + "=" * 60)
    print(f"🚀 {name}")
    print("=" * 60)

    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
    )

    if result.returncode != 0:
        print(f"\n❌ {name} failed.")
        sys.exit(result.returncode)

    print(f"✅ {name} completed successfully.")


def main():
    run_step(
        "Step 1 — Parse Zeek logs",
        [
            sys.executable,
            "src/zeek/parse_conn.py",
        ],
    )

    run_step(
        "Step 2 — Detect threats",
        [
            sys.executable,
            "src/zeek/detect_flows.py",
        ],
    )

    run_step(
        "Step 3 — Send detections to SentinelAI",
        [
            sys.executable,
            "src/zeek/send_to_api.py",
        ],
    )

    print("\n" + "=" * 60)
    print("🛡️ SentinelAI Zeek Pipeline Completed")
    print("=" * 60)
    print("PCAP → Zeek → Parse → Detect → FastAPI → Incidents")


if __name__ == "__main__":
    main()