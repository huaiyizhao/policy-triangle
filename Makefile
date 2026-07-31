LATEX=pdflatex

.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib figures/policy_triangle.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.toc
