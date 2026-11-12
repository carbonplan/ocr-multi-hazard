#!/usr/bin/env bash
# One `fdp_train.py task` on a Coiled VM, from the scripts `launch` staged on S3.
# Usage: fdp_task.sh <code prefix> <task arguments...>
set -euo pipefail
code=$1
shift
mkdir -p fdp
cd fdp
python -c 'import s3fs, sys; s3fs.S3FileSystem().get(sys.argv[1] + "/", ".", recursive=True)' "$code"
exec python fdp_train.py task "$@"
