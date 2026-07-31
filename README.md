# The Policy Triangle paper

This directory contains a concise arXiv-oriented LaTeX manuscript for the policy-triangle framework developed in the accompanying discussion. The revised paper has six main sections and a broader, source-checked bibliography.

## Files

- `main.tex` — complete English manuscript
- `references.bib` — bibliography
- `figures/policy_triangle.tex` — deterministic TikZ figure based on the three-color triangular reference layout
- `figures/imagegen-prompt.txt` — prepared prompt for an optional generated graphical abstract
- `docs/index.md` — editable Markdown source for the GitHub Pages technical note
- `docs/_layouts/technical.html` — Jekyll page shell
- `docs/assets/technical.css` — technical-note styling
- `docs/assets/policy-triangle.svg` — editable web figure
- `Makefile` — local build commands

## Build

With a normal TeX Live installation:

```bash
make
```

or directly:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The manuscript has been compiled with Tectonic and visually inspected page by page. The author is Huaiyi Zhao; GPT drafting and language-editing assistance is disclosed in the title-page footnote.

## Blog

The publishable web article lives in `docs/`. Edit `docs/index.md`; GitHub Pages
uses Jekyll to generate the final HTML automatically. With Jekyll installed,
preview locally with:

    jekyll serve --source docs

For GitHub Pages, select **Deploy from a branch** and use the repository's
`/docs` directory as the publishing source.

## Submission scope

The manuscript is deliberately framed as a conceptual taxonomy paper. Its central contribution is the unified comparison of existing mechanisms, not a new optimizer or convergence theorem. It contains no invented empirical results. A stronger conference submission should add controlled experiments that separately instrument `B`, `U`, and `E` under backend mismatch, policy staleness, and multiple optimizer epochs.
