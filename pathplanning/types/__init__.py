from typing import Tuple, Union

import numpy as np

Int = Union[np.int32, np.int64, int]
Float = Union[np.float32, float]
Number = Union[Int, Float]
Position = Tuple[Number, Number, Number]
LenPath = Tuple[Number, "dubins._DubinsPath"]

__all__ = ["Int", "Float", "Number", "Position", "LenPath"]
