#!/usr/bin/env bash
# Starts Prometheus and Grafana on this computer, for the monitoring demo (no Docker needed).
#
#   brew install prometheus grafana            # once
#   uvicorn app:app                            # the service, on port 8000 (in another terminal)
#   bash monitoring/run_local.sh               # this script
#   python monitoring/send_traffic.py          # some traffic (in another terminal)
#
# Then open Grafana at http://localhost:3000 (no login) and Prometheus at http://localhost:9090.
# The numbers are kept in a temporary folder that is deleted when you stop the script (Ctrl+C).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
mkdir -p "$work/prometheus" "$work/grafana/logs" "$work/provisioning/datasources" "$work/provisioning/dashboards"

# Grafana reads its data source and the dashboard from files, so nothing is clicked together.
cp "$here/grafana-datasource.local.yml" "$work/provisioning/datasources/datasource.yml"
cat > "$work/provisioning/dashboards/dashboards.yml" <<YAML
apiVersion: 1
providers:
  - name: product-similarity
    type: file
    options:
      path: $here
YAML

trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$work"' EXIT

prometheus \
  --config.file="$here/prometheus.local.yml" \
  --storage.tsdb.path="$work/prometheus" \
  --web.listen-address=127.0.0.1:9090 > "$work/prometheus.log" 2>&1 &

GF_PATHS_DATA="$work/grafana" \
GF_PATHS_LOGS="$work/grafana/logs" \
GF_PATHS_PLUGINS="$work/grafana/plugins" \
GF_PATHS_PROVISIONING="$work/provisioning" \
GF_AUTH_ANONYMOUS_ENABLED=true \
GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
GF_SERVER_HTTP_ADDR=127.0.0.1 \
GF_ANALYTICS_REPORTING_ENABLED=false \
GF_ANALYTICS_CHECK_FOR_UPDATES=false \
grafana server --homepath "$(brew --prefix grafana)/share/grafana" > "$work/grafana.log" 2>&1 &

echo "Prometheus: http://localhost:9090   Grafana: http://localhost:3000   (Ctrl+C to stop)"
wait
