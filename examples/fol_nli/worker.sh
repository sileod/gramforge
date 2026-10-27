#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python -u -m gramforge.fol_nli.build "$@"
