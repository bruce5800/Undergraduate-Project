# Build paper/main.pdf with pdflatex + bibtex.
# Targets:
#   make          → build PDF
#   make clean    → remove aux files
#   make distclean→ also remove PDF

MAIN = main
SECTIONS = $(wildcard sections/*.tex)
BIB = references.bib

.PHONY: all clean distclean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(SECTIONS) $(BIB)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.fls *.fdb_latexmk \
	      sections/*.aux

distclean: clean
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	open $(MAIN).pdf
