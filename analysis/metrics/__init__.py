from analysis.metrics.spatiotemporal import WalkingSpeed, StrideLength, StrideTime, Cadence
from analysis.metrics.variability import StrideTimeCV, StrideLengthCV
from analysis.metrics.kinematics import KneeROM, HipROM

ALL_SCALAR_METRICS: list = [
    WalkingSpeed(),
    StrideLength(),
    StrideTime(),
    Cadence(),
    StrideTimeCV(),
    StrideLengthCV(),
    KneeROM(),
    HipROM(),
]
