# Background reference requirements

- dataset language: `fr`
- status: `not_provided`
- path: `not provided`
- SHA-256: `not available`
- source label: `missing_metadata`
- accepted input: JSON `{word: count}` or CSV `word,count`
- validation: non-empty; finite nonnegative counts; positive total mass
- reader: `braindecoding.evaluation.ovmi.load_reference_distribution`

不会自动下载、选择或将 Wikipedia/OpenSubtitles/SUBTLEX/随机 corpus 指定为正式 background。

missing_metadata: source_label, version, license, language, tokenization
