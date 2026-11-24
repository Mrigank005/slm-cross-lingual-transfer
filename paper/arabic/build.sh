#!/bin/sh
# Renders the Arabic snippets used in Table (appendix examples). Requires LuaLaTeX
# and a font with Arabic glyphs (Times New Roman on Windows; change Path= elsewhere).
for f in ar*.tex; do lualatex -interaction=nonstopmode "$f" >/dev/null; done
rm -f *.aux *.log
