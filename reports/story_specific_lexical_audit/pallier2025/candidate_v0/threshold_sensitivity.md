# Minimum train material occurrence sensitivity

以下 eligible type count 只来自 train material counts。由于没有 background，Top50 story-specific ranking、threshold 间 overlap/Jaccard 均未计算。

| min train count | eligible types | Top20 | Top50 | Top100 | Top150 | Top50 ranking |
|---:|---:|---|---|---|---|---|
| 1 | 2020 | True | True | True | True | blocked: background_reference_not_provided |
| 2 | 880 | True | True | True | True | blocked: background_reference_not_provided |
| 5 | 340 | True | True | True | True | blocked: background_reference_not_provided |
| 10 | 179 | True | True | True | True | blocked: background_reference_not_provided |
| 20 | 92 | True | True | False | False | blocked: background_reference_not_provided |
