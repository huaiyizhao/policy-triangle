# A Unified Estimator View blog

This directory is a standalone GitHub Pages site for:

> **A Unified Estimator View of Policy-Mismatch Mitigation in LLM Reinforcement Learning**

## Edit the article

Edit **index.md** for all technical content. The other files are separated by
responsibility:

- **_layouts/technical.html** — page shell, navigation, metadata, and MathJax
- **assets/technical.css** — restrained technical styling
- **assets/policy-mismatch-intro.svg** — accessible opening illustration
- **assets/policy-triangle-hero.svg** — compact ratio-placement figure
- **assets/is-tail-operators.svg** — estimator statistic, mask, and weight choices
- **assets/estimator-decision-flow.svg** — practical estimator decision flow
- **assets/bias-variance-tradeoff.svg** — appendix-level statistical intuition

GitHub Pages runs Jekyll automatically. If Jekyll is installed locally, preview
from the repository root with:

    jekyll serve --source docs

## Publish with GitHub Pages

1. Push the repository to GitHub.
2. Open **Settings → Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Select the target branch and the **/docs** folder.
5. Save. GitHub will compile **docs/index.md** into the public index page.

The site presents the complete work as a standalone technical blog. Math
formulas use MathJax from jsDelivr; figures are responsive SVG assets. No PDF
download or LaTeX source archive is published on the website.
