# 编译中文版 main.pdf（xelatex 替换 pdflatex 以支持中文字体）。
# 目标：
#   make          → 构建 PDF
#   make clean    → 清理 aux 文件
#   make distclean→ 同时清理 PDF
#   make view     → 编译并打开 PDF

MAIN = main
SECTIONS = $(wildcard sections/*.tex)
BIB = references.bib

.PHONY: all clean distclean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(SECTIONS) $(BIB)
	xelatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	xelatex -interaction=nonstopmode $(MAIN).tex
	xelatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.fls *.fdb_latexmk \
	      sections/*.aux

distclean: clean
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	open $(MAIN).pdf
