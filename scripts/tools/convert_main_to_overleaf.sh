#!/bin/sh
set -eu

input_tex=${1:-doc/paper/main.tex}

case "$input_tex" in
    *.tex) output_tex=${2:-"${input_tex%.tex}-overleaf.tex"} ;;
    *) output_tex=${2:-"${input_tex}-overleaf.tex"} ;;
esac

if [ ! -f "$input_tex" ]; then
    printf 'Input file not found: %s\n' "$input_tex" >&2
    exit 1
fi

tmp_file=$(mktemp)
cleanup() {
    rm -f "$tmp_file"
}
trap cleanup EXIT INT TERM

perl -0pe '
    s!^[ \t]*\\graphicspath\s*\{[^\n]*\}!\\graphicspath{{fig/}}!mg;
    s!(\\includegraphics\*?(?:\[[^\]]*\])?\{)(?:\.\./\.\./figs/|(?:\./)?figs/|fig/)([^}]*)\}!$1$2}!g;
' "$input_tex" > "$tmp_file"

mv "$tmp_file" "$output_tex"
printf 'Wrote %s\n' "$output_tex"
