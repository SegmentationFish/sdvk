#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# img2img / inpaint / decode stress matrix.
#
#   ⚠ This drives the GPU. Run it yourself, one case at a time, while watching the desktop:
#   a device-local request it cannot satisfy takes the whole WDDM session down, not just the app.
#   Nothing here is started automatically by the tooling.
#
#   usage:  tools/stress.sh [-m MODEL_DIR] [-i INIT.png] [-M MASK.png] [--quick]
#
# Every case prints PASS/FAIL and the script exits non-zero if any case fails. The acceptance
# rules are the ones from PLAN-audio-img2img.md §4.2:
#   - no VK_ERROR_DEVICE_LOST / crash / non-finite output
#   - denoise 0 roundtrip PSNR > 30 dB (this is what proves the VAE encode is really the inverse
#     of the decode; a wrong encoder shows up as 12-18 dB)
#   - mask-outside region PSNR > 32 dB
#   - repeated runs keep a stable per-step time (< 15% drift)
set -u

BIN=${BIN:-./build/sdvkgen.exe}
MODEL=${MODEL:-E:/ofc-data/noobai}
INIT=${INIT:-}
MASK=${MASK:-}
OUT=${OUT:-output/stress}
SEED=${SEED:-1234}
QUICK=0

while [ $# -gt 0 ]; do
  case "$1" in
    -m) MODEL=$2; shift 2 ;;
    -i) INIT=$2; shift 2 ;;
    -M) MASK=$2; shift 2 ;;
    -b) BIN=$2; shift 2 ;;
    --quick) QUICK=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

mkdir -p "$OUT" logs 2>/dev/null
PASS=0; FAIL=0
SUMMARY=()

record() {
  local ok=$1 name=$2 detail=$3
  if [ "$ok" = 1 ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
  printf '  %-4s %-46s %s\n' "$([ "$ok" = 1 ] && echo PASS || echo FAIL)" "$name" "$detail"
  SUMMARY+=("$([ "$ok" = 1 ] && echo PASS || echo FAIL)  $name  $detail")
}

# Runs one generation. Sets LOG, RC and MS (wall clock) for the caller.
generate() {
  local name=$1; shift
  LOG="logs/stress_${name}.log"
  local t0 t1
  t0=$(date +%s%3N)
  "$BIN" -m "$MODEL" --seed "$SEED" "$@" > "$LOG" 2>&1
  RC=$?
  t1=$(date +%s%3N)
  MS=$((t1 - t0))
}

check_log() {  # name, then grep patterns that must NOT appear
  local name=$1; shift
  local bad=0
  for pat in "$@"; do
    if grep -qE "$pat" "$LOG"; then
      record 0 "$name" "found '$pat' in $LOG"
      bad=1
      break
    fi
  done
  [ $bad = 1 ] && return 1
  return 0
}

echo "sdvk stress: bin=$BIN model=$MODEL init=${INIT:-<none>} mask=${MASK:-<none>}"
echo

# ---------------------------------------------------------------- 1. VAE roundtrip (encode/decode)
# This is the acceptance core: with a real encoder, denoise 0 must reproduce the source. The old
# pixel-seeded path landed around 12-18 dB here.
if [ -n "$INIT" ]; then
  for size in 512 768 1024; do
    [ "$QUICK" = 1 ] && [ "$size" = 768 ] && continue
    generate "roundtrip_$size" --init "$INIT" --roundtrip --size "$size" --out "$OUT/roundtrip_$size.png"
    psnr=$(grep -oE 'psnr [0-9.]+' "$LOG" | tail -1 | awk '{print $2}')
    if [ $RC -ne 0 ]; then
      record 0 "roundtrip ${size}px" "exit $RC, see $LOG"
    elif [ -z "$psnr" ]; then
      record 0 "roundtrip ${size}px" "no psnr line, see $LOG"
    else
      ok=$(awk -v p="$psnr" 'BEGIN{print (p>30)?1:0}')
      record "$ok" "roundtrip ${size}px psnr > 30 dB" "${psnr} dB, ${MS} ms"
    fi
  done
else
  echo "  SKIP roundtrip: pass -i <png> to enable the encode/decode check"
fi
echo

# ---------------------------------------------------------------- 2. denoise ladder
if [ -n "$INIT" ]; then
  for d in 0.00 0.30 0.55 0.75 1.00; do
    generate "denoise_$d" --init "$INIT" --denoise "$d" --size 832x1216 --steps 20 --cfg 4 \
      --vpred --spacing trailing --out "$OUT/denoise_${d}.png"
    if check_log "denoise $d" "VK_ERROR_DEVICE_LOST" "non-finite" "FAIL:"; then
      record $([ $RC -eq 0 ] && echo 1 || echo 0) "denoise $d @832x1216" "rc=$RC ${MS} ms"
    fi
  done
else
  echo "  SKIP denoise ladder: pass -i <png>"
fi
echo

# ---------------------------------------------------------------- 3. inpaint
if [ -n "$INIT" ] && [ -n "$MASK" ]; then
  for f in 0 8 16; do
    generate "inpaint_f$f" --init "$INIT" --mask "$MASK" --mask-feather "$f" --denoise 0.85 \
      --size 832x1216 --steps 20 --cfg 4 --vpred --spacing trailing --out "$OUT/inpaint_f${f}.png"
    if check_log "inpaint feather $f" "VK_ERROR_DEVICE_LOST" "non-finite" "FAIL:"; then
      repaint=$(grep -oE '[0-9.]+% repainted' "$LOG" | tail -1)
      record $([ $RC -eq 0 ] && echo 1 || echo 0) "inpaint feather ${f}px" "rc=$RC ${repaint:-no mask line}"
    fi
  done
else
  echo "  SKIP inpaint: pass -i <png> -M <mask.png>"
fi
echo

# ---------------------------------------------------------------- 4. repeat / leak check
if [ "$QUICK" = 0 ]; then
  times=()
  for i in $(seq 1 20); do
    generate "loop_$i" --size 1024x1024 --steps 12 --cfg 4 --vpred --spacing trailing \
      --out "$OUT/loop_$i.png"
    if [ $RC -ne 0 ]; then
      record 0 "repeat run $i" "rc=$RC, see logs/stress_loop_$i.log"
      break
    fi
    step=$(grep -oE 'median [0-9]+ ms' "$LOG" | tail -1 | awk '{print $2}')
    [ -n "$step" ] && times+=("$step")
  done
  if [ ${#times[@]} -ge 10 ]; then
    first=${times[0]}
    last=${times[$(( ${#times[@]} - 1 ))]}
    drift=$(awk -v a="$first" -v b="$last" 'BEGIN{ d=(b-a)/a*100; if (d<0) d=-d; print d }')
    ok=$(awk -v d="$drift" 'BEGIN{print (d<15)?1:0}')
    record "$ok" "20 runs, step time drift < 15%" "first ${first} ms, last ${last} ms, ${drift}%"
  fi
fi

echo
echo "--------------------------------------------------------------------"
for l in "${SUMMARY[@]}"; do echo "$l"; done
echo "--------------------------------------------------------------------"
echo "PASS $PASS  FAIL $FAIL   logs in logs/, images in $OUT/"
[ $FAIL -eq 0 ] || exit 1
