#!/bin/bash

set -e

echo "=========================================="
echo "🛡️ SentinelAI Automated Zeek Pipeline"
echo "=========================================="

echo ""
echo "🚀 Step 1 — Running Zeek in Docker"

docker compose run --rm zeek

echo "✅ Zeek completed"

echo ""
echo "🚀 Step 2 — Parsing Zeek logs"

python src/zeek/parse_conn.py

echo "✅ Parsing completed"

echo ""
echo "🚀 Step 3 — Detecting threats"

python src/zeek/detect_flows.py

echo "✅ Threat detection completed"

echo ""
echo "🚀 Step 4 — Sending detections to SentinelAI"

python src/zeek/send_to_api.py

echo "✅ API integration completed"

echo ""
echo "=========================================="
echo "🛡️ SentinelAI Pipeline Completed"
echo "=========================================="
echo "PCAP → Docker Zeek → Parse → Detect → API"
echo "=========================================="
