# Policy Triangle blog

This directory is a standalone GitHub Pages site for:

> **The Policy Triangle: A Taxonomy of Policy Mismatch in LLM Reinforcement Learning**

## Preview locally

From the paper repository, run:

    python3 -m http.server 8000 --directory docs

Then open <http://localhost:8000>.

## Publish with GitHub Pages

1. Push the repository to GitHub.
2. Open **Settings → Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Select the target branch and the **/docs** folder.
5. Save. GitHub will publish **docs/index.html**.

The page is framework-free and requires no build step. Math formulas use
MathJax from jsDelivr; all layout and the policy-triangle figure are embedded
in **index.html**.

