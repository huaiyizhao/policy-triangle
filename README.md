# A Unified Estimator View blog

This repository publishes a technical blog about a unified estimator framework
for comparing methods that mitigate policy mismatch in LLM reinforcement
learning. The policy triangle is retained as an auxiliary ratio-placement map.

## Blog files

- `docs/index.md` — editable Markdown source for the complete article
- `docs/_layouts/technical.html` — Jekyll page shell
- `docs/assets/technical.css` — responsive technical-blog styling
- `docs/assets/policy-mismatch-intro.svg` — opening illustration
- `docs/assets/policy-triangle-hero.svg` — compact ratio-placement figure
- `docs/assets/is-tail-operators.svg` — estimator statistic, mask, and weight choices
- `docs/assets/bias-variance-tradeoff.svg` — appendix-level statistical intuition

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

The article offers a unified estimator view of the mitigation solution space.
Its contribution is a shared estimator template and conditional decision
process for comparing existing mechanisms, not a new optimizer, convergence
theorem, or empirical performance claim.
