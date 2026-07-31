# The Policy Triangle blog

This repository publishes a technical blog about the policy-triangle framework
for comparing policy mismatch mechanisms in LLM reinforcement learning.

## Blog files

- `docs/index.md` — editable Markdown source for the complete article
- `docs/_layouts/technical.html` — Jekyll page shell
- `docs/assets/technical.css` — responsive technical-blog styling
- `docs/assets/policy-mismatch-intro.svg` — opening illustration
- `docs/assets/policy-triangle-hero.svg` — detailed taxonomy figure

## Preview

With Jekyll installed:

```bash
jekyll serve --source docs
```

## Publish

For GitHub Pages, select **Deploy from a branch** and use the repository's
`/docs` directory as the publishing source. The site intentionally publishes no
PDF download.

## Scope

The article is a conceptual taxonomy. Its contribution is a common coordinate
system for comparing existing mechanisms, not a new optimizer, convergence
theorem, or empirical performance claim.
