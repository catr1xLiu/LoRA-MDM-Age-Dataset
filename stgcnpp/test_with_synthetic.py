"""Generate a synthetic NTU RGB+D .skeleton file and run it through STGCN++.

The .skeleton file format (from the official MATLAB reader):
    <num_frames>
    for each frame:
        <num_bodies>
        for each body:
            <bodyID> <clippedEdges> <handLeftConf> <handLeftState>
            <handRightConf> <handRightState> <isRestricted> <leanX> <leanY> <trackingState>
            <num_joints>   (always 25)
            for each joint:
                <x> <y> <z> <depthX> <depthY> <colorX> <colorY>
                <orientW> <orientX> <orientY> <orientZ> <trackingState>

3D joint ordering (0-indexed) in NTU RGB+D:
  0  SpineBase   1  SpineMid    2  Neck         3  Head
  4  ShoulderL   5  ElbowL      6  WristL       7  HandL
  8  ShoulderR   9  ElbowR     10  WristR       11  HandR
 12  HipL       13  KneeL      14  AnkleL       15  FootL
 16  HipR       17  KneeR      18  AnkleR       19  FootR
 20  SpineShoulder  21  HandTipL  22  ThumbL
                    23  HandTipR  24  ThumbR
"""
import math
import pickle
import struct
import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1. Define a T-pose reference skeleton (in meters, camera-centred coords)
#    Camera is ~2 m in front of the subject; y is up.
# ---------------------------------------------------------------------------
# fmt: off
T_POSE = np.array([
    # id  joint name          x       y       z
    [ 0,  0.00,  0.00,  2.00],   # 0  SpineBase
    [ 0,  0.00,  0.25,  2.00],   # 1  SpineMid
    [ 0,  0.00,  0.65,  2.00],   # 2  Neck
    [ 0,  0.00,  0.80,  2.00],   # 3  Head
    [ 0, -0.20,  0.55,  2.00],   # 4  ShoulderL
    [ 0, -0.40,  0.55,  2.00],   # 5  ElbowL
    [ 0, -0.60,  0.55,  2.00],   # 6  WristL
    [ 0, -0.68,  0.55,  2.00],   # 7  HandL
    [ 0,  0.20,  0.55,  2.00],   # 8  ShoulderR
    [ 0,  0.40,  0.55,  2.00],   # 9  ElbowR
    [ 0,  0.60,  0.55,  2.00],   # 10 WristR
    [ 0,  0.68,  0.55,  2.00],   # 11 HandR
    [ 0, -0.10, -0.05,  2.00],   # 12 HipL
    [ 0, -0.10, -0.45,  2.00],   # 13 KneeL
    [ 0, -0.10, -0.85,  2.00],   # 14 AnkleL
    [ 0, -0.10, -0.98,  1.90],   # 15 FootL
    [ 0,  0.10, -0.05,  2.00],   # 16 HipR
    [ 0,  0.10, -0.45,  2.00],   # 17 KneeR
    [ 0,  0.10, -0.85,  2.00],   # 18 AnkleR
    [ 0,  0.10, -0.98,  1.90],   # 19 FootR
    [ 0,  0.00,  0.50,  2.00],   # 20 SpineShoulder
    [ 0, -0.74,  0.55,  2.00],   # 21 HandTipL
    [ 0, -0.70,  0.57,  2.00],   # 22 ThumbL
    [ 0,  0.74,  0.55,  2.00],   # 23 HandTipR
    [ 0,  0.70,  0.57,  2.00],   # 24 ThumbR
], dtype=np.float32)             # shape (25, 4) — first col unused
# fmt: on

JOINTS_XYZ = T_POSE[:, 1:]     # (25, 3)
NUM_JOINTS = 25


def make_action_sequence(num_frames: int = 150, action: str = 'wave') -> np.ndarray:
    """Return joint positions (num_frames, 25, 3) for a simple action."""
    frames = np.tile(JOINTS_XYZ[None], (num_frames, 1, 1)).copy()
    t = np.linspace(0, 2 * math.pi, num_frames)

    if action == 'wave':
        # Right arm waves up and down
        amp = 0.4
        for f in range(num_frames):
            angle = amp * math.sin(t[f])
            # Rotate right shoulder joint chain around shoulder
            sh = frames[f, 8].copy()
            for j in [9, 10, 11, 23, 24]:
                d = frames[f, j] - sh
                frames[f, j, 1] = sh[1] + d[0] * math.sin(angle) + d[1] * math.cos(angle)
    elif action == 'squat':
        for f in range(num_frames):
            drop = 0.15 * (1 - math.cos(t[f]))  # 0..0.3 m
            for j in [0, 1, 12, 13, 16, 17, 20]:
                frames[f, j, 1] -= drop
    else:
        # Add small random jitter (standing still)
        frames += np.random.randn(*frames.shape).astype(np.float32) * 0.005

    return frames  # (T, 25, 3)


# ---------------------------------------------------------------------------
# 2. Write a .skeleton file
# ---------------------------------------------------------------------------

def write_skeleton_file(path: str, seq: np.ndarray, num_persons: int = 1):
    """Write NTU .skeleton text file from (T, 25, 3) or (M, T, 25, 3) array."""
    if seq.ndim == 3:
        seq = seq[np.newaxis]                     # (1, T, 25, 3)
    M, T, V, C = seq.shape
    assert V == 25 and C == 3

    with open(path, 'w') as f:
        f.write(f'{T}\n')
        for frame_idx in range(T):
            f.write(f'{M}\n')
            for person_idx in range(M):
                # body header: bodyID clippedEdges hLC hLS hRC hRS isR leanX leanY trackingState
                f.write(f'{person_idx + 1} 0 1 0 1 0 0 0.00 0.00 2\n')
                f.write(f'{V}\n')
                for j in range(V):
                    x, y, z = seq[person_idx, frame_idx, j]
                    # depthX depthY colorX colorY  orientW orientX orientY orientZ  trackingState
                    f.write(
                        f'{x:.6f} {y:.6f} {z:.6f} '
                        f'0.500000 0.500000 640.000000 360.000000 '
                        f'1.000000 0.000000 0.000000 0.000000 2\n'
                    )


# ---------------------------------------------------------------------------
# 3. Parse a .skeleton file → (M, T, 25, 3)
# ---------------------------------------------------------------------------

def parse_skeleton_file(path: str) -> np.ndarray:
    """Parse NTU .skeleton file; returns (M, T, 25, 3) float32 array.

    Uses the max persons detected across all frames and zero-pads missing ones.
    """
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    idx = 0
    num_frames = int(lines[idx]); idx += 1
    all_frames = []

    for _ in range(num_frames):
        num_bodies = int(lines[idx]); idx += 1
        bodies = {}
        for b in range(num_bodies):
            parts = lines[idx].split(); idx += 1
            body_id = int(parts[0])
            num_joints = int(lines[idx]); idx += 1
            joints = np.zeros((num_joints, 3), dtype=np.float32)
            for j in range(num_joints):
                vals = list(map(float, lines[idx].split())); idx += 1
                joints[j] = vals[:3]   # x, y, z only
            bodies[body_id] = joints
        all_frames.append(bodies)

    # Collect unique person IDs in order of first appearance
    person_ids = []
    for frame in all_frames:
        for pid in frame:
            if pid not in person_ids:
                person_ids.append(pid)

    M = len(person_ids)
    T = num_frames
    seq = np.zeros((M, T, 25, 3), dtype=np.float32)
    for t, frame in enumerate(all_frames):
        for m, pid in enumerate(person_ids):
            if pid in frame:
                seq[m, t] = frame[pid]

    return seq   # (M, T, 25, 3)


# ---------------------------------------------------------------------------
# 4. Preprocess: PreNormalize3D (centre on spine base) → uniform sample
# ---------------------------------------------------------------------------

def pre_normalize_3d(seq: np.ndarray) -> np.ndarray:
    """Subtract spine-base (joint 0) of person 0 to centre the skeleton.

    Matches pyskl's PreNormalize3D transform.
    """
    # Use mean position of spine-base across frames as the origin
    origin = seq[0, :, 0, :]              # (T, 3)
    origin_mean = origin.mean(axis=0)     # (3,)
    seq = seq - origin_mean[None, None, None, :]
    return seq


def uniform_sample(seq: np.ndarray, clip_len: int = 100) -> np.ndarray:
    """Uniformly sample clip_len frames from (M, T, 25, 3)."""
    T = seq.shape[1]
    indices = np.linspace(0, T - 1, clip_len, dtype=int)
    return seq[:, indices]                # (M, clip_len, 25, 3)


def format_for_model(seq: np.ndarray, max_persons: int = 2) -> torch.Tensor:
    """Pad/crop to max_persons and return (1, M, T, 25, 3) tensor."""
    M = seq.shape[0]
    if M < max_persons:
        pad = np.zeros((max_persons - M, *seq.shape[1:]), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)
    else:
        seq = seq[:max_persons]
    # shape: (M, T, 25, 3) → (1, M, T, 25, 3)
    return torch.from_numpy(seq).unsqueeze(0)


# ---------------------------------------------------------------------------
# 5. NTU RGB+D action labels (120 classes — 1-indexed in the dataset)
# ---------------------------------------------------------------------------

NTU120_LABELS = {
    1: "drink water", 2: "eat meal/snack", 3: "brushing teeth", 4: "brushing hair",
    5: "drop", 6: "pickup", 7: "throw", 8: "sitting down", 9: "standing up",
    10: "clapping", 11: "reading", 12: "writing", 13: "tear up paper",
    14: "wear jacket", 15: "take off jacket", 16: "wear a shoe", 17: "take off a shoe",
    18: "wear on glasses", 19: "take off glasses", 20: "put on a hat/cap",
    21: "take off a hat/cap", 22: "cheer up", 23: "hand waving", 24: "kicking something",
    25: "reach into pocket", 26: "hopping (one foot jumping)", 27: "jump up",
    28: "make a phone call/answer phone", 29: "playing with phone/tablet",
    30: "typing on a keyboard", 31: "pointing to something with finger",
    32: "taking a selfie", 33: "check time (from watch)", 34: "rub two hands together",
    35: "nod head/bow", 36: "shake head", 37: "wipe face", 38: "salute",
    39: "put the palms together", 40: "cross hands in front (scare/stop)",
    41: "sneeze/cough", 42: "staggering", 43: "falling", 44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)", 46: "touch back (backache)",
    47: "touch neck (neckache)", 48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper)/feeling warm",
    50: "punching/slapping other person", 51: "kicking other person",
    52: "pushing other person", 53: "pat on back of other person",
    54: "point finger at the other person", 55: "hugging other person",
    56: "giving something to other person", 57: "touch other person's pocket",
    58: "handshaking", 59: "walking towards each other",
    60: "walking apart from each other",
    # NTU120 extras:
    61: "put on headphone", 62: "take off headphone", 63: "shoot at the basket",
    64: "bounce ball", 65: "tennis bat swing", 66: "juggling table tennis balls",
    67: "hush (quite)", 68: "flick hair", 69: "thumb up", 70: "thumb down",
    71: "make ok sign", 72: "make victory sign", 73: "staple book",
    74: "counting money", 75: "cutting nails", 76: "cutting paper (using scissors)",
    77: "snapping fingers", 78: "open bottle", 79: "sniff (smell)", 80: "squat down",
    81: "toss a coin", 82: "fold paper", 83: "ball up paper",
    84: "play magic cube", 85: "apply cream on face", 86: "apply cream on hand back",
    87: "put on bag", 88: "take off bag", 89: "put something into a bag",
    90: "take something out of a bag", 91: "open a box", 92: "move heavy objects",
    93: "shake fist", 94: "throw up cap/hat", 95: "hands up (both arms)",
    96: "cross arms", 97: "arm circles", 98: "arm swings",
    99: "running on the spot", 100: "butt kicks (kick backward)",
    101: "cross toe touch", 102: "side kick", 103: "yawn",
    104: "stretch oneself", 105: "blow nose",
    106: "hit other person with something", 107: "wield knife towards other person",
    108: "knock over other person (hit with body)", 109: "grab other person's stuff",
    110: "shoot at other person with a gun", 111: "step on foot",
    112: "high-five", 113: "cheers and drink", 114: "carry something with other person",
    115: "take a photo of other person", 116: "follow other person",
    117: "whisper in other person's ear", 118: "exchange things with other person",
    119: "support somebody with hand", 120: "finger-guessing game (playing rock-paper-scissors)",
}


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from model import STGCNPP

    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'stgcnpp_ntu120_3dkp_joint.pth')
    skel_path = '/tmp/synthetic_ntu.skeleton'

    print('=' * 60)
    print('STGCN++ end-to-end test with synthetic NTU skeleton data')
    print('=' * 60)

    # --- Generate and write synthetic skeleton ---
    for action_name in ['wave', 'squat']:
        print(f'\n--- Action: {action_name} ---')
        seq = make_action_sequence(num_frames=150, action=action_name)  # (T, 25, 3)
        write_skeleton_file(skel_path, seq, num_persons=1)
        print(f'  Written {skel_path}  ({150} frames, 1 person)')

        # --- Parse it back ---
        parsed = parse_skeleton_file(skel_path)  # (M, T, 25, 3)
        print(f'  Parsed  shape: {parsed.shape}')
        print(f'  Joint 0 (spine base) range: x∈[{parsed[0,:,0,0].min():.3f}, {parsed[0,:,0,0].max():.3f}]')

        # --- Preprocess ---
        normed = pre_normalize_3d(parsed)
        sampled = uniform_sample(normed, clip_len=100)
        tensor = format_for_model(sampled, max_persons=2)  # (1, 2, 100, 25, 3)
        print(f'  Model input shape: {tuple(tensor.shape)}')

        # --- Load model and run ---
        model = STGCNPP(num_classes=120, pretrained=ckpt_path)
        model.eval()
        with torch.no_grad():
            logits = model(tensor)              # (1, 120)

        pred_class = logits.argmax(dim=1).item()   # 0-indexed
        confidence = logits.softmax(dim=1)[0, pred_class].item()
        label_name = NTU120_LABELS.get(pred_class + 1, f'class_{pred_class}')

        print(f'  Predicted class : {pred_class} → "{label_name}"')
        print(f'  Confidence      : {confidence:.3f}')

        # Top-5
        top5 = logits.softmax(dim=1)[0].topk(5)
        print('  Top-5:')
        for score, idx in zip(top5.values.tolist(), top5.indices.tolist()):
            print(f'    [{idx:3d}] {NTU120_LABELS.get(idx+1, "?"):40s}  {score:.4f}')


if __name__ == '__main__':
    main()
