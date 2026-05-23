# Compilar o relatório
# Requer: tectonic (brew install tectonic)
# Alternativa: MacTeX/TeX Live com pdflatex
.PHONY: pdf pdf-pdflatex clean

pdf:
	tectonic -X compile relatorio.tex

pdf-pdflatex:
	pdflatex -interaction=nonstopmode relatorio.tex
	pdflatex -interaction=nonstopmode relatorio.tex

clean:
	rm -f relatorio.aux relatorio.log relatorio.out relatorio.toc relatorio.pdf
