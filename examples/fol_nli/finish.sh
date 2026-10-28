#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false
pool=$1
output=$2
groups=${3:-100000}
shards=${4:-3}
while true; do
    ready=true
    for ((shard=0; shard<shards; shard++)); do
        if [[ ! -f "$pool/complete-shard$shard.json" ]]; then ready=false; fi
    done
    if "$ready"; then break; fi
    sleep 60
done
python -m gramforge.fol_nli.build --output "$pool" --groups "$groups" --finalize
python -m gramforge.fol_nli.inspect "$pool"
python -m gramforge.fol_nli.curate --input "$pool" --output "$output" --size 100000
python -m gramforge.fol_nli.verify "$output" --pool "$pool" --workers 24 --independent-sample 1000
