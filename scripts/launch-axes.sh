#!/usr/bin/env bash
# Launch one Modal GPU lab per mood axis and write its task spec.
#   scripts/launch-axes.sh tender:cold playful:grave lush:plain
# Each lab's URL lands in wsp_runs/<run>/LAB_URL; task specs in steer/wsp/tasks/<run>.md.
# Stop them all afterwards with:  scripts/with-modal uv run modal app stop -y <app-id>  (see `modal app list`)
cd "$(dirname "$0")/.."
for pair in "$@"; do
  c=${pair%%:*}; a=${pair##*:}; run="$c-$a"
  mkdir -p "wsp_runs/$run"; rm -f "wsp_runs/$run/LAB_URL"
  nohup scripts/with-modal uv run modal run steer/wsp/modal_lab.py --run "$run" --budget 15 > "wsp_runs/$run/modal.log" 2>&1 &
  echo "$run: launching (pid $!)"
done
for pair in "$@"; do
  c=${pair%%:*}; a=${pair##*:}; run="$c-$a"
  for i in $(seq 1 120); do [ -s "wsp_runs/$run/LAB_URL" ] && break; sleep 5; done
  url=$(cat "wsp_runs/$run/LAB_URL" 2>/dev/null)
  if [ -z "$url" ]; then echo "$run: NO LAB_URL (see wsp_runs/$run/modal.log)"; continue; fi
  uv run python -m steer.wsp.make_task "$c" "$a" --run "$run" --budget 15 --lab-url "$url" >/dev/null
  WSP_LAB=$url uv run wsp status | python3 -c "import json,sys;d=json.load(sys.stdin);print('$run', 'ready:', d['passes_left'], 'passes at', '$url')"
done
