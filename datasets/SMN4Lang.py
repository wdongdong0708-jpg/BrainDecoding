"""兼容旧导入；正式实现位于 ``braindecoding.data.smn4lang``。"""

from braindecoding.data.smn4lang import *  # noqa: F401,F403
from braindecoding.data.smn4lang import (
    _configured_split_by_run,
    _load_alignment,
    _open_processed_recording,
    _parse_bool,
    _preprocessing_signature,
    _read_raw_fif,
    _recording_files,
    _repository_gpt2_paths,
    _scale_channels,
    _sha256_text,
    _signature_digest,
    _training_event_digest,
    _training_vocabulary,
    _valid_recording_cache,
)
