from typing import Final

FPS: Final[int] = 20
DT: Final[float] = 1.0 / FPS
N_GAIT_PTS: Final[int] = 101
MIN_CLIP_FRAMES: Final[int] = 20

# HumanML3D 22-joint indices
PELVIS = 0
L_HIP,     R_HIP     = 1,  2
SPINE1                = 3
L_KNEE,    R_KNEE     = 4,  5
SPINE2                = 6
L_ANKLE,   R_ANKLE    = 7,  8
SPINE3                = 9
L_FOOT,    R_FOOT     = 10, 11
NECK                  = 12
L_COLLAR,  R_COLLAR   = 13, 14
HEAD                  = 15
L_SHOULDER,R_SHOULDER = 16, 17
L_ELBOW,   R_ELBOW    = 18, 19
L_WRIST,   R_WRIST    = 20, 21

JOINT_NAMES: Final[list[str]] = [
    "pelvis", "L_hip", "R_hip", "spine1",
    "L_knee", "R_knee", "spine2", "L_ankle",
    "R_ankle", "spine3", "L_foot", "R_foot",
    "neck", "L_collar", "R_collar", "head",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist",
]

HEATMAP_ORDER: Final[list[int]] = [
    R_WRIST, L_WRIST, R_ELBOW, L_ELBOW,
    R_SHOULDER, L_SHOULDER, HEAD, R_COLLAR, L_COLLAR, NECK,
    R_FOOT, L_FOOT, SPINE3, R_ANKLE, L_ANKLE, SPINE2,
    R_KNEE, L_KNEE, SPINE1, R_HIP, L_HIP, PELVIS,
]
HEATMAP_LABELS: Final[list[str]] = [JOINT_NAMES[i] for i in HEATMAP_ORDER]

AGE_BINS: Final[dict[str, tuple[int, int]]] = {
    "young": (21, 40),
    "old":   (65, 100),
}

GROUP_COLORS: Final[dict[str, str]] = {
    "young": "#2196F3",
    "old":   "#F44336",
}

# Representative ages used for generated clips (midpoint of each bin)
GROUP_REPRESENTATIVE_AGE: Final[dict[str, float]] = {
    "young": 30.0,
    "old":   75.0,
}
