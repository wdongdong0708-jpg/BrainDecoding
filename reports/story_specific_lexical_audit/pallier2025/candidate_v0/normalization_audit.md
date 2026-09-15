# Normalization audit

- dataset: `Pallier2025 Little Prince`
- language: `fr`
- normalization: `strip_lower_preserve_french_accents_and_tokenization`
- full story equals frozen story_reference: `true`

故事与 background 均只做 `strip` + `lower`；保留法语重音和 source tokenization。不合并 `j + avais`，不 lemmatize、accent-strip、stem 或 clitic-merge。

异常 token 只加 diagnostic flags，不会自动删除。
