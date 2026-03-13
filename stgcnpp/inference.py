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
 12  HipL       13  KneeL     14  AnkleL       15  FootL
 16  HipR       17  KneeR     18  AnkleR       19  FootR
 20  SpineShoulder  21  HandTipL  22  ThumbL
                    23  HandTipR  24  ThumbR
"""

import torch
from torch.utils.data import DataLoader

from dataset import SkeletonDataset
from model import STGCNPP

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

BATCH_SIZE = 16
NUM_WORKERS = 4
PRINT_EVERY = 500


def run_inference():
    # Device
    DEVICE = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {DEVICE}")

    # Dataset & DataLoader - use ALL annotations (same as old code)
    dataset = SkeletonDataset(data_path, clip_len=100, max_person=2, normalize=True)
    total_clips = len(dataset)
    print(f"Running inference on {total_clips} clips (full dataset)")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type != "cpu"),
        shuffle=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    # Model
    model: STGCNPP = STGCNPP(pretrained="checkpoints/stgcnpp_ntu120_3dkp_joint.pth")
    model = model.to(DEVICE)
    model.eval()
    model.freeze_backbone()

    # Counters
    correct_count: int = 0
    top5_correct_count: int = 0
    processed: int = 0

    from tqdm import tqdm

    with torch.no_grad():
        for keypoints, labels in tqdm(loader, desc="Inference"):
            keypoints = keypoints.to(DEVICE)  # (N, 2, 100, 25, 3)
            labels = labels.to(DEVICE)  # (N,)

            output = model(keypoints)  # (N, 120)
            probs = torch.softmax(output, dim=1)

            _, top5_indices = torch.topk(probs, k=5, dim=1)  # (N, 5)
            top1_indices = top5_indices[:, 0]  # (N,)

            correct_count += int((top1_indices == labels).sum())
            top5_correct_count += int(
                (top5_indices == labels.unsqueeze(1)).any(dim=1).sum()
            )

            processed += keypoints.size(0)

            # Print progress every PRINT_EVERY batches
            if processed // BATCH_SIZE % PRINT_EVERY == 0:
                current_acc = correct_count / processed * 100
                print(
                    f"Processed {processed}/{total_clips} clips. "
                    f"Current Top-1: {current_acc:.1f}%"
                )

    print(
        f"\nOut of total {total_clips} tested clips, "
        f"{correct_count} are correctly identified (top-1), "
        f"{top5_correct_count} are in top-5 results."
    )
    print(f"Top-1 accuracy: {correct_count / total_clips * 100:.2f}%")
    print(f"Top-5 accuracy: {top5_correct_count / total_clips * 100:.2f}%")


if __name__ == "__main__":
    run_inference()
