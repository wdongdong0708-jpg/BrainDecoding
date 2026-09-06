# SMN4Lang word decoding

This task learns from all usable Mandarin word events in the preprocessed MEG
derivative of OpenNeuro `ds004078` version `1.2.1`, and evaluates retrieval on
a frozen 50-word vocabulary.

Protocol:

- Local subset: `sub-01`, 60 natural-story runs.
- Split unit: complete story run.
- Train/validation/test: runs `1-50` / `51-55` / `56-60`.
- Training supervision: every non-empty word with a complete 3-second window;
  words outside the top 50 are no longer discarded.
- Evaluation vocabulary: the 50 most frequent non-empty words in train runs
  only, with lexical tie-breaking. Test MEG is not used for its selection.
- Context groups: true sentence boundaries from the sentence-per-line public
  `scripts/story_*.txt` files, with a hard fallback chunk limit of 128 usable
  words. Physical batches never exceed the configured batch size.
- Window: `[word onset, word onset + 3 s)` at 50 Hz, restricted to each run's
  measured `Beg`/`End` support.
- Timing: published fMRI-clock alignment minus `10.65 s`, plus the run's MEG
  `Beg` trigger and `39.5 ms` acoustic delivery delay.
- Text target: the dataset-provided 1,024-D word-level GPT-2 representation at
  final layer 24. Contextual occurrences are averaged within train runs only to
  form one frozen prototype per word; validation/test text does not define
  candidates. Layer 24 is explicit because the audited middle layers are highly
  anisotropic for this fixed-vocabulary retrieval task.
- Selection metric: macro Top-10 retrieval over the observed members of the
  frozen 50-word vocabulary.
- Optimization budget: exactly 6,400 successful optimizer updates. The cosine
  schedule has a 12,800-update horizon so the final learning rate matches the
  halfway point reached by the audited LibriBrain run; 100 epochs is only a
  safety ceiling and early stopping cannot fire before update 6,400.

Build and audit the event table without opening any FIF signal payload:

```powershell
D:\soft\miniconda\envs\bm5060\python.exe run.py word_decoding SMN4Lang prepare
```

Materialize train/validation inputs explicitly:

```powershell
D:\soft\miniconda\envs\bm5060\python.exe run.py word_decoding SMN4Lang prepare --with-text-embeddings --with-meg-cache
```

The default SMN4Lang config is now `configs/SMN4Lang_gpt2.yaml`. The earlier
Mengzi-T5 Transformer experiment remains reproducible with
`--config configs/SMN4Lang.yaml`; its caches and outputs are not overwritten.

The 60 large FIF files are processed recording by recording and MEG channels
are loaded in small chunks. This stage can be slow; an existing valid cache is
reused. Test runs are excluded unless `--include-test` is provided explicitly.

Train and evaluate validation data:

```powershell
D:\soft\miniconda\envs\bm5060\python.exe run.py word_decoding SMN4Lang train
D:\soft\miniconda\envs\bm5060\python.exe run.py word_decoding SMN4Lang evaluate --split val
```

The completed Mengzi-T5 convolution-only ablation remains available with:

```powershell
D:\soft\miniconda\envs\bm5060\python.exe run.py word_decoding SMN4Lang train --config configs/SMN4Lang_conv_only.yaml
D:\soft\miniconda\envs\bm5060\python.exe run.py word_decoding SMN4Lang evaluate --config configs/SMN4Lang_conv_only.yaml --split val
```

Test evaluation is an explicit separate action:

```powershell
D:\soft\miniconda\envs\bm5060\python.exe run.py word_decoding SMN4Lang evaluate --split test
```
