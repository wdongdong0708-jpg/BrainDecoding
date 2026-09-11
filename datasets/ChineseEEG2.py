"""兼容旧导入；正式实现位于 ``braindecoding.data.chineseeeg2``。"""

from braindecoding.data.chineseeeg2 import *  # noqa: F401,F403
from braindecoding.data.chineseeeg2 import (
    _直接BrainVision记录,
    _打开处理后记录,
    _构建单一实际朗读事件表,
    _签名摘要,
    _缩放通道,
    _记录缓存有效,
    _读取脑电记录,
    _预处理签名,
)
