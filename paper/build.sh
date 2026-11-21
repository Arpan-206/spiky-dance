#!/bin/sh
# Regenerate every figure and numbers.tex from the episode folders, then typeset.
# VLMA_EPISODE=<clip_id> ./build.sh points everything at another episode.
set -e
cd "$(dirname "$0")"
PY="../.venv/bin/python"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/vlma-mpl"
for f in figures/fig_*.py; do (cd figures && "../$PY" "$(basename "$f")"); done
(cd figures && "../$PY" make_numbers.py > /dev/null)
export PATH=/Library/TeX/texbin:$PATH
latexmk -pdf -jobname=vlma -outdir=build -interaction=nonstopmode -halt-on-error main.tex
cp build/vlma.pdf vlma.pdf
