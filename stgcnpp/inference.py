"""
Each pickle file corresponds to an action recognition dataset. The content of a pickle file is a dictionary with two fields: split and annotations


Split:
The value of the split field is a dictionary: the keys are the split names, while the values are lists of video identifiers that belong to the specific clip.


Annotations:
The value of the annotations field is a list of skeleton annotations, each skeleton annotation is a dictionary, containing the following fields:

 - frame_dir (str): The identifier of the corresponding video.

 - total_frames (int): The number of frames in this video.

 - img_shape (tuple[int]): The shape of a video frame, a tuple with two elements, in the format of (height, width). Only required for 2D skeletons.

 - original_shape (tuple[int]): Same as img_shape.

 - label (int): The action label.

 - keypoint (np.ndarray, with shape [M x T x V x C]): The keypoint annotation.
    - M: number of persons;
    - T: number of frames (same as total_frames);
    - V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. );
    - C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).

 - keypoint_score (np.ndarray, with shape [M x T x V]): The confidence score of keypoints. Only required for 2D skeletons.

3D joint ordering (0-indexed) in NTU RGB+D:
  0  SpineBase   1  SpineMid    2  Neck         3  Head
  4  ShoulderL   5  ElbowL      6  WristL       7  HandL
  8  ShoulderR   9  ElbowR     10  WristR       11  HandR
 12  HipL       13  KneeL      14  AnkleL       15  FootL
 16  HipR       17  KneeR      18  AnkleR       19  FootR
 20  SpineShoulder  21  HandTipL  22  ThumbL
                    23  HandTipR  24  ThumbR
"""

import pickle

import numpy as np
import torch

from graph import _NTU_NEIGHBOR_BASE
from model import STGCNPP
from utils import _angle_between, _rotation_matrix

# Convert from 1-indexed to 0-indexed bone pairs
NTU_PAIRS = [(i - 1, j - 1) for i, j in _NTU_NEIGHBOR_BASE]

data_path = "data/ntu120_3danno.pkl"

# fmt: off
NTU_ACTION_NAMES = {
    # --- Daily Actions (A1–A40, A61–A102) ---
    0: "drink water",           1: "eat meal",              2: "brush teeth",
    3: "brush hair",            4: "drop",                  5: "pick up",
    6: "throw",                 7: "sit down",              8: "stand up",
    9: "clapping",              10: "reading",              11: "writing",
    12: "tear up paper",        13: "put on jacket",        14: "take off jacket",
    15: "put on a shoe",        16: "take off a shoe",      17: "put on glasses",
    18: "take off glasses",     19: "put on a hat/cap",     20: "take off a hat/cap",
    21: "cheer up",             22: "hand waving",          23: "kicking something",
    24: "reach into pocket",    25: "hopping",              26: "jump up",
    27: "phone call",           28: "play with phone",      29: "type on keyboard",
    30: "point to something",   31: "taking a selfie",      32: "check time (watch)",
    33: "rub two hands",        34: "nod head/bow",         35: "shake head",
    36: "wipe face",            37: "salute",               38: "put palms together",
    39: "cross hands in front",
    # --- Medical Conditions (A41–A49) ---
    40: "sneeze/cough",         41: "staggering",           42: "falling down",
    43: "headache",             44: "chest pain",           45: "back pain",
    46: "neck pain",            47: "nausea/vomiting",      48: "fan self",
    # --- Mutual Actions (A50–A60) ---
    49: "punch/slap",           50: "kicking",              51: "pushing",
    52: "pat on back",          53: "point finger",         54: "hugging",
    55: "giving object",        56: "touch pocket",         57: "shaking hands",
    58: "walking towards",      59: "walking apart",
    # --- Daily Actions cont. (A61–A102) ---
    60: "put on headphone",     61: "take off headphone",   62: "shoot at basket",
    63: "bounce ball",          64: "tennis bat swing",     65: "juggle table tennis ball",
    66: "hush",                 67: "flick hair",           68: "thumb up",
    69: "thumb down",           70: "make OK sign",         71: "make victory sign",
    72: "staple book",          73: "counting money",       74: "cutting nails",
    75: "cutting paper",        76: "snap fingers",         77: "open bottle",
    78: "sniff/smell",          79: "squat down",           80: "toss a coin",
    81: "fold paper",           82: "ball up paper",        83: "play magic cube",
    84: "apply cream on face",  85: "apply cream on hand",  86: "put on bag",
    87: "take off bag",         88: "put object into bag",  89: "take object out of bag",
    90: "open a box",           91: "move heavy objects",   92: "shake fist",
    93: "throw up cap/hat",     94: "capitulate",           95: "cross arms",
    96: "arm circles",          97: "arm swings",           98: "run on the spot",
    99: "butt kicks",           100: "cross toe touch",     101: "side kick",
    # --- Medical Conditions cont. (A103–A105) ---
    102: "yawn",                103: "stretch oneself",     104: "blow nose",
    # --- Mutual Actions cont. (A106–A120) ---
    105: "hit with object",     106: "wield knife",         107: "knock over",
    108: "grab stuff",          109: "shoot with gun",      110: "step on foot",
    111: "high-five",           112: "cheers and drink",    113: "carry object",
    114: "take a photo",        115: "follow",              116: "whisper",
    117: "exchange things",     118: "support somebody",    119: "rock-paper-scissors",
}
# fmt: on


def normalize_skeleton(keypoint):
    """Reproduce pyskl PreNormalize3D.

    Input:  np.ndarray (M, T, V, C) — raw keypoints from the annotation pkl
    Output: np.ndarray (M, T', V, C) — centred, aligned, zero-frames stripped

    Three differences from the old version:
      1. Centres on joint 1 (SpineMid), not joint 0
      2. Aligns spine (joint 0→1) to Z-axis [0,0,1], then
         shoulders (joint 8→4) to X-axis [1,0,0] via Rodrigues rotation
      3. Strips all-zero frames and masks zero joints
    """
    keypoint = keypoint.copy().astype(np.float32)
    M, T, V, C = keypoint.shape

    if keypoint.sum() == 0:
        return keypoint

    # 1. Strip all-zero frames; if 2 persons, keep the one with more valid frames as person 0
    index0 = [i for i in range(T) if not np.allclose(keypoint[0, i], 0, atol=1e-5)]

    if M == 2:
        index1 = [i for i in range(T) if not np.allclose(keypoint[1, i], 0, atol=1e-5)]
        if len(index0) < len(index1):
            keypoint = keypoint[:, index1]
            keypoint = keypoint[[1, 0]]  # swap so the "main" person is index 0
        else:
            keypoint = keypoint[:, index0]
    else:
        keypoint = keypoint[:, index0]

    # 2. Centre on joint 1 (SpineMid) of person 0, frame 0
    main_body_center = keypoint[0, 0, 1].copy()
    mask = ((keypoint != 0).sum(-1) > 0)[
        ..., None
    ]  # (M, T', V, 1) — keep zero joints at zero
    keypoint = (keypoint - main_body_center) * mask

    # 3. Rotate spine vector (joint 0 → joint 1) onto Z-axis [0, 0, 1]
    spine = keypoint[0, 0, 1] - keypoint[0, 0, 0]
    axis_z = np.cross(spine, [0, 0, 1])
    angle_z = _angle_between(spine, [0, 0, 1])
    rot_z = _rotation_matrix(axis_z, angle_z)
    keypoint = np.einsum("mtvc,kc->mtvk", keypoint, rot_z)

    # 4. Rotate shoulder vector (joint 8 → joint 4) onto X-axis [1, 0, 0]
    shoulder = keypoint[0, 0, 8] - keypoint[0, 0, 4]
    axis_x = np.cross(shoulder, [1, 0, 0])
    angle_x = _angle_between(shoulder, [1, 0, 0])
    rot_x = _rotation_matrix(axis_x, angle_x)
    keypoint = np.einsum("mtvc,kc->mtvk", keypoint, rot_x)

    return keypoint


def uniform_sample(keypoint, clip_len=100):
    """Uniformly resample to exactly clip_len frames along the T axis.

    Input:  np.ndarray (M, T, V, C)
    Output: np.ndarray (M, clip_len, V, C)
    """
    T = keypoint.shape[1]
    if T == clip_len:
        return keypoint
    indices = np.linspace(0, T - 1, clip_len).astype(int)
    return keypoint[:, indices]


# Load dataset
with open(data_path, "rb") as f:
    data = pickle.load(f)

# Load model
DEVICE = torch.device("cuda:0")
model: STGCNPP = STGCNPP(pretrained="checkpoints/stgcnpp_ntu120_3dkp_joint.pth")
model = model.to(DEVICE)
model.eval()
model.freeze_backbone()

# Counters
total_clips = len(data["annotations"])
total_clips = 5000
correct_count: int = 0
top5_correct_count: int = 0

# Go through all clips
for clip in range(total_clips):
    # Extract anotation and label
    anotation = data["annotations"][clip]
    label: int = int(anotation["label"])

    # Conver keypoint to expected format by model
    # Model expects (N, M, T, V, C) where N = 1 (1 clip), M = 2 (two skeletons)

    keypoint_np = anotation["keypoint"]
    keypoint_np = normalize_skeleton(keypoint_np)
    keypoint_np = uniform_sample(keypoint_np)
    keypoint_raw = torch.tensor(keypoint_np, dtype=torch.float32)
    if keypoint_raw.shape[0] == 1:
        padding = torch.zeros_like(keypoint_raw)  # (1, T, V, C)
        keypoint_raw = torch.cat([keypoint_raw, padding], dim=0)  # (2, T, V, C)
    keypoint = keypoint_raw.unsqueeze(0)  # shape: (1, 2, T, V, C)
    keypoint = keypoint.to(DEVICE)

    # Run inference using GPU
    with torch.no_grad():
        output = model(keypoint)

    # Convert confidence to probability using SoftMax function
    probs = torch.softmax(output, dim=1)

    # Take top 10 results
    values, indies = torch.topk(probs, k=10, dim=1)
    values, indies = values.squeeze(), indies.squeeze()

    top_k_correct = -1  # None
    for k in range(5):
        if indies[k] == label:
            top_k_correct = k
            top5_correct_count += 1
            break

    correct_count += int(indies[0] == label)

    index: int = int(indies[0])
    result_action_name = (
        NTU_ACTION_NAMES[index] if index in NTU_ACTION_NAMES else "<< Unknown Action >>"
    )
    label_action_name = (
        NTU_ACTION_NAMES[label] if label in NTU_ACTION_NAMES else "<< Unknown Action >>"
    )
    percent_conf = int(values[0] * 100)

    print(
        f"Tested clip {clip + 1} / {total_clips}. Label: {label_action_name}, Prediction: {result_action_name} ({percent_conf})."
    )

print(
    f"Out of total {total_clips} tested clips, {correct_count} are correctly identified, {top5_correct_count} are in top 5 results."
)
