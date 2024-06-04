'''Type definitions for max-plus algebra'''

from fractions import Fraction
from typing import List, Union

TTimeStamp = Union[Fraction,None]
TTimeStampList = List[TTimeStamp]
TTimeStampFloat = Union[float,None]
TTimeStampFloatList = List[TTimeStampFloat]
TMPVector = TTimeStampList
TMPVectorList = List[TMPVector]
TMPMatrix = TMPVectorList
