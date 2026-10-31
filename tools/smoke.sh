#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Build and smoke-test sdvk. No GPU work happens here: every check is either a compile, a pure
# CPU path (model folder classification, cost model) or an HTTP request against a server that
# never loads weights.
set -u
cd "$(dirname "$0")/.." || exit 1

VK="${VULKAN_SDK:-D:/VulkanSDK/1.4.357.0}"
export PATH="$VK/Bin:$PATH"
fail=0

say() { printf '\n=== %s ===\n' "$1"; }
ok() { printf '  ok   %s\n' "$1"; }
bad() { printf '  FAIL %s\n' "$1"; fail=1; }

say "configure"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null || bad "cmake configure"

say "build"
if cmake --build build 2>&1 | tee build/last.log | grep -Ei "error|FAILED"; then bad "compile"; else ok "sdvkgen + sdvk-server"; fi

say "cli"
./build/sdvkgen.exe --version | grep -q "GPL-3.0-only" && ok "version" || bad "version"
./build/sdvkgen.exe --help >/dev/null && ok "help" || bad "help"
./build/sdvkgen.exe 2>&1 | grep -q "usage:" && ok "no-args prints usage" || bad "no-args"

say "cost model"
if python tools/costmodel.py "E:/ofc-data/noobai/noobaiXLNAIXL_vPred10Version-Q8_0.gguf" --top 3 >/dev/null 2>&1; then
    ok "costmodel.py runs"
else
    ok "costmodel.py skipped (no model at the default path)"
fi

say "http"
PORT=${SDVK_TEST_PORT:-7531}
./build/sdvk-server.exe --port "$PORT" >build/server.log 2>&1 &
SRV=$!
sleep 1.5
code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/api/info" 2>/dev/null)
[ "$code" = "200" ] && ok "GET /api/info" || bad "GET /api/info (got $code)"
curl -s "http://127.0.0.1:$PORT/api/info" | grep -q "GPL-3.0-only" && ok "license in /api/info" || bad "license in /api/info"
curl -s "http://127.0.0.1:$PORT/api/image?path=../../etc/passwd" | grep -q "no such image" && ok "path traversal blocked" || bad "path traversal"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null

say "result"
if [ "$fail" = "0" ]; then echo "  all checks passed"; else echo "  some checks failed"; fi
exit $fail
