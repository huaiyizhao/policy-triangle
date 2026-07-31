# Policy Triangle blog

This directory is a standalone GitHub Pages site for:

> **The Policy Triangle: A Taxonomy of Policy Mismatch in LLM Reinforcement Learning**

## Edit the article

Edit **index.md** for all technical content. The other files are separated by
responsibility:

- **_layouts/technical.html** — page shell, navigation, metadata, and MathJax
- **assets/technical.css** — restrained technical styling
- **assets/policy-triangle.svg** — editable web figure
- **policy-triangle.pdf** — downloadable paper

GitHub Pages runs Jekyll automatically. If Jekyll is installed locally, preview
from the repository root with:

    jekyll serve --source docs

## Publish with GitHub Pages

1. Push the repository to GitHub.
2. Open **Settings → Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Select the target branch and the **/docs** folder.
5. Save. GitHub will compile **docs/index.md** into the public index page.

The article presents the full technical note rather than a promotional landing
page. Math formulas use MathJax from jsDelivr. Only the PDF is offered for
download; the LaTeX source archive is not published on the website.
