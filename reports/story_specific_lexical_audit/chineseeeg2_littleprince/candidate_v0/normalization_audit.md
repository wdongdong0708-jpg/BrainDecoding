# Normalization audit

- dataset: `ChineseEEG2 Little Prince`
- language: `zh`
- normalization: `existing_event_table_standard_word_strip_only`
- full story equals frozen story_reference: `true`

故事侧直接使用 frozen event table 的 `标准词/normalized_word`。background 只做 `strip`；不重新分词、不做简繁转换、词干化、同义词或概念合并。

异常 token 只加 diagnostic flags，不会自动删除。
