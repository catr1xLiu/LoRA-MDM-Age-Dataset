import numpy as np

# ── Helpers (matching pyskl PreNormalize3D exactly) ──────────────────────


def _unit_vector(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def _angle_between(v1, v2):
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0.0
    return np.arccos(np.clip(np.dot(_unit_vector(v1), _unit_vector(v2)), -1.0, 1.0))


def _rotation_matrix(axis, theta):
    """Rodrigues rotation (quaternion form) — identical to pyskl."""
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3, dtype=np.float32)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ],
        dtype=np.float32,
    )
