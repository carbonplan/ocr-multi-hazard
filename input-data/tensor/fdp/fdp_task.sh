#!/usr/bin/env bash
# One FDP script on a Coiled VM, from the copies `fdp.launch` staged on S3.
# Usage: fdp_task.sh <code prefix> <script> <arguments...>
set -euo pipefail
code=$1
shift
mkdir -p fdp
cd fdp
python -c 'import s3fs, sys; s3fs.S3FileSystem().get(sys.argv[1] + "/", ".", recursive=True)' "$code"
exec python "$@"
