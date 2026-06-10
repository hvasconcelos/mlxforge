# Build the mlxforge technical white paper.
#
#   make        build the PDF (latexmk: pdflatex + bibtex as needed)
#   make clean  remove LaTeX build artifacts (keeps the PDF)
#   make distclean  remove artifacts and the PDF

DOC := mlxforge-whitepaper

.PHONY: all clean distclean

all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex references.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(DOC).tex

clean:
	latexmk -c $(DOC).tex
	rm -f $(DOC).bbl $(DOC).blg

distclean:
	latexmk -C $(DOC).tex
	rm -f $(DOC).bbl $(DOC).blg
