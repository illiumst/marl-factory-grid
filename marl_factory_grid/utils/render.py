from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RenderEntity:
    name: str
    pos: np.array
    value: float = 1
    value_operation: str = 'none'
    state: str = None
    id: int = 0
    aux: Any = None
    real_name: str = 'none'
