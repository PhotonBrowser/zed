#!/bin/sh
set -eu

zed_root=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
uid=$(id -u)
run_id=$$
service="com.openai.codex.photon.metalprototype"
channel="standalone-view-${uid}-${run_id}"
plist="${TMPDIR:-/tmp}/photon-presentation-${uid}-${run_id}.plist"
service_err="${TMPDIR:-/tmp}/photon-presentation-${uid}-${run_id}.err"
domain="gui/${uid}"
producer_pid=

cleanup() {
    status=$?
    if [ -n "$producer_pid" ]; then
        kill "$producer_pid" >/dev/null 2>&1 || true
        wait "$producer_pid" >/dev/null 2>&1 || true
    fi
    launchctl bootout "$domain/$service" >/dev/null 2>&1 || true
    if [ "$status" -ne 0 ] && [ -f "$service_err" ]; then
        cat "$service_err"
    fi
    rm -f "$plist" "$service_err"
}
trap cleanup EXIT INT TERM

cd "$zed_root"
cargo build -p gpui_apple --features test-support \
    --example presentation_xpc_service --example presentation_xpc_smoke

service_binary="$zed_root/target/debug/examples/presentation_xpc_service"
cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>${service}</string>
<key>ProgramArguments</key><array><string>${service_binary}</string><string>${service}</string></array>
<key>MachServices</key><dict>
  <key>${service}</key><true/>
  <key>${service}.iosurface</key><true/>
</dict>
<key>RunAtLoad</key><true/>
<key>StandardErrorPath</key><string>${service_err}</string>
</dict></plist>
EOF

launchctl bootout "$domain/$service" >/dev/null 2>&1 || true
launchctl bootstrap "$domain" "$plist"
sleep 1
PHOTON_PRESENTATION_XPC_SERVICE="$service" \
    "$zed_root/target/debug/examples/presentation_xpc_smoke" producer "$channel" &
producer_pid=$!
PHOTON_PRESENTATION_XPC_SERVICE="$service" \
    "$zed_root/target/debug/examples/presentation_xpc_smoke" consumer "$channel"
wait "$producer_pid"

printf 'GPUI capture: %s/gpui-crossprocess-iosurface.png\n' "${TMPDIR:-/tmp}"
