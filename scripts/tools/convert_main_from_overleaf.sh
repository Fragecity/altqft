#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
figs_root="$repo_root/figs"

input_tex=${1:-doc/paper/main-overleaf.tex}

case "$input_tex" in
    *.tex)
        default_output="${input_tex%-overleaf.tex}.tex"
        if [ "$default_output" = "$input_tex.tex" ]; then
            default_output="${input_tex%.tex}.local.tex"
        fi
        output_tex=${2:-"$default_output"}
        ;;
    *)
        output_tex=${2:-"${input_tex}.local"} ;;
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

FIGS_ROOT="$figs_root" perl -0MFile::Find -pe '
    my $figs_root = $ENV{FIGS_ROOT};
    my %resolved_cache;

    sub resolve_graphic_path {
        my ($path) = @_;

        $path =~ s!^(?:fig/|(?:\./)?figs/|\.\./\.\./figs/)!!;
        return $path if $path =~ m!/!;

        return $resolved_cache{$path} if exists $resolved_cache{$path};

        my @matches;
        find(
            sub {
                return unless -f $_;
                return unless $_ eq $path;
                my $relative = $File::Find::name;
                $relative =~ s!^\Q$figs_root\E/!!;
                push @matches, $relative;
            },
            $figs_root
        );

        $resolved_cache{$path} = @matches == 1 ? $matches[0] : $path;
        return $resolved_cache{$path};
    }

    s!^[ \t]*\\graphicspath\s*\{[^\n]*\}!\\graphicspath{{../../figs/}}!mg;
    s!(\\includegraphics\*?(?:\[[^\]]*\])?\{)([^}]*)(\})!$1 . resolve_graphic_path($2) . $3!ge;
' "$input_tex" > "$tmp_file"

mv "$tmp_file" "$output_tex"
printf 'Wrote %s\n' "$output_tex"
