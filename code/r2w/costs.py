"""R2W 冻结成本口径。

算法工件中的成本单位是可审计的 ``word_count``，不等同于任意模型的
tokenizer token 或供应商账单 token。真实 API 用量另由 ``runtime_usage``
记录，二者不得混写。
"""

from __future__ import annotations

import numpy as np

from .representations import Repr
from .text import word_count

COST_UNIT = "word_count"


def representation_costs(representation: Repr) -> np.ndarray:
    """返回 [write, index, reader_payload] 的冻结三分量成本。"""
    payload = float(word_count(representation.payload))
    keys = float(sum(word_count(value) for value in representation.keys))
    # raw 没有生成式草稿写入；其原始正文仍需进入索引并在命中时给 reader。
    write = 0.0 if representation.arm == "raw" else payload + keys
    return np.asarray((write, payload + keys, payload), dtype=np.float32)
