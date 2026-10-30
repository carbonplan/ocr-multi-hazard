#!/bin/bash
# Download the public inputs of chaz_downscale.py and chaz_tropicona.py into
# ~/.cache/ocr/chaz-local (or $CHAZ_LOCAL_DIR), checking md5 sums and skipping
# files already present:
#   zenodo/     CHAZ tutorial data, ERA5 2017 forcing (Zenodo 11631930, CC BY 4.0)
#   tropicona/  CMIP6 CHAZ tracks from the TROPICONA tutorial (Zenodo 12533388, CC BY 4.0)
set -euo pipefail
root=${CHAZ_LOCAL_DIR:-$HOME/.cache/ocr/chaz-local}

fetch_one() {
  url=$1; key=$2; sum=${3#md5:}; dest=$4
  out="$dest/$key"
  if [ -f "$out" ] && [ "$(md5 -q "$out")" = "$sum" ]; then echo "ok   $key"; return; fi
  for try in 1 2 3; do
    curl -sL --retry 3 -o "$out.part" "$url" && [ "$(md5 -q "$out.part")" = "$sum" ] \
      && mv "$out.part" "$out" && echo "got  $key" && return
    echo "retry $try $key"
  done
  echo "FAIL $key"
}
export -f fetch_one

for pair in 11631930:zenodo 12533388:tropicona; do
  id=${pair%%:*}; dest=$root/${pair#*:}
  mkdir -p "$dest"
  curl -sL "https://zenodo.org/api/records/$id" \
    | python3 -c "
import json, sys
for f in json.load(sys.stdin)['files']:
    print(f['links']['self'], f['key'], f['checksum'])" \
    | awk -v d="$dest" '{print $1, $2, $3, d}' \
    | xargs -P 3 -n 4 bash -c 'fetch_one "$@"' _
done
