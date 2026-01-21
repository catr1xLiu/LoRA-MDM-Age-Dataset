#!/usr/bin/env python3
"""
Van Criekinge → processed markers for SMPL fitting (and HumanML3D later)

- Reads C3D (mm → m)
- Sanitizes trial names early (e.g., "SUBJ1 (0)" → "SUBJ1_0")
- Keeps only 3D position markers (drops computed angle channels)
- Optional gap filling (linear, per marker/axis, with max gap length)
- Optional resampling to target FPS (default 20 Hz, inte1_dataset_rpolation)
- Writes:
    <out>/<SUBJXX>/<TRIAL>_markers_full.npz         # original (post basic filtering)
    <out>/<SUBJXX>/<TRIAL>_markers_positions.npz    # position-only, gapfilled+resampled
    <out>/<SUBJXX>/<TRIAL>_metadata.json
    <out>/<SUBJXX>/<TRIAL>_mosh_config.json         # kept for compatibility if needed
"""

import sys
# Import metadata from the local data directory
sys.path.insert(0, "./data")

from van_criekinge_unprocessed_1.metadata import create_able_bodied_metadata, create_stroke_metadata

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import c3d

# -------- logging ----------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------- small utils ----------
def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "")

def _linear_fill_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Fill NaN gaps with linear interpolation for 1D array, only if gap length <= max_gap.
    Longer gaps remain NaN.
    """
    y = x.copy()
    n = len(y)
    isn = np.isnan(y)
    if not isn.any():
        return y

    idx = np.arange(n)
    valid = ~isn
    if valid.sum() < 2:
        return y  # not enough points to interpolate

    # interpolate across all NaNs first
    y_interp = np.interp(idx, idx[valid], y[valid])

    # only commit interpolated values where gap length <= max_gap
    i = 0
    while i < n:
        if isn[i]:
            j = i
            while j < n and isn[j]:
                j += 1
            gap_len = j - i
            if gap_len <= max_gap and i > 0 and j < n:
                y[i:j] = y_interp[i:j]
            i = j
        else:
            i += 1
    return y

def _resample_series(y: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """
    Resample a 1D series y defined on t_src onto t_tgt using linear interpolation,
    ignoring NaNs (requires >=2 valid points).
    """
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return np.full_like(t_tgt, np.nan, dtype=float)
    return np.interp(t_tgt, t_src[valid], y[valid])

# -------- main preprocessor ----------
class VanCriekingeDatasetPreprocessor:
    def __init__(self,
                 base_data_dir: str,
                 output_dir: str,
                 min_valid_ratio: float = 0.5,
                 gap_fill_max: int = 10,
                 resample_fps: Optional[float] = 20.0,
                 resample_mode: str = "interp"  # "interp" | "decimate"
                 ):
        """
        Args:
            base_data_dir: raw dataset root
            output_dir: where to write processed files
            min_valid_ratio: frame kept if >= this ratio of marker coords are valid
            gap_fill_max: maximum consecutive NaN length to fill (in frames) for positions
            resample_fps: if not None, resample positions to this FPS (default 20.0)
            resample_mode: "interp" (time-accurate) or "decimate" (every k-th frame)
        """
        self.base_data_dir = Path(base_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_valid_ratio = float(min_valid_ratio)
        self.gap_fill_max = int(gap_fill_max)
        self.resample_fps = resample_fps
        assert resample_mode in ("interp", "decimate")
        self.resample_mode = resample_mode

        # Subject metadata
        self.able_bodied_metadata = create_able_bodied_metadata()
        self.stroke_metadata      = create_stroke_metadata()

    # ------------ IO ------------
    def load_c3d_file(self, filepath: Path) -> Optional[Dict]:
        """Read C3D, mm→m, basic frame filtering."""
        try:
            with open(filepath, 'rb') as f:
                reader = c3d.Reader(f)
                header = reader.header
                first_frame = header.first_frame
                last_frame = header.last_frame
                frame_rate = float(header.frame_rate)

                frames = []
                for _, points, _ in reader.read_frames():
                    pts = points[:, :3].astype(np.float64)  # (N,3)
                    pts[pts < -999] = np.nan
                    frames.append(pts)

            if not frames:
                logger.warning(f"No frames in {filepath.name}")
                return None

            marker_data = np.stack(frames, axis=0)  # (T,N,3)
            # marker labels (best-effort)
            marker_names = []
            try:
                if hasattr(reader, 'point_labels'):
                    marker_names = [s.strip() for s in reader.point_labels if s.strip()]
            except Exception:
                pass
            if not marker_names or len(marker_names) != marker_data.shape[1]:
                marker_names = [f"marker_{i:02d}" for i in range(marker_data.shape[1])]

            # mm → m
            marker_data = marker_data / 1000.0

            # Keep frames with enough valid data
            valid_ratio = np.mean(~np.isnan(marker_data), axis=(1, 2))
            keep = valid_ratio >= self.min_valid_ratio
            if keep.sum() == 0:
                logger.warning(f"All frames invalid in {filepath.name}")
                return None

            marker_data = marker_data[keep]
            n_frames = marker_data.shape[0]

            logger.info(f"Loaded {filepath.name}: {n_frames} frames, {marker_data.shape[1]} markers @ {frame_rate:.2f} Hz")
            return dict(
                marker_data=marker_data,
                marker_names=marker_names,
                frame_rate=frame_rate,
                n_frames=n_frames,
                first_frame=int(first_frame),
                last_frame=int(last_frame),
                units="meters",
                file_path=str(filepath),
                valid_marker_ratio=float(np.mean(~np.isnan(marker_data)))
            )
        except Exception as e:
            logger.error(f"Load failed for {filepath}: {e}")
            return None

    # ------------ dataset traversal ------------
    def process_able_bodied_subject(self, subject_id: str) -> List[Dict]:
        subject_dir = self.base_data_dir / "able_bodied" / subject_id.upper()
        if not subject_dir.exists():
            logger.warning(f"Missing subject dir: {subject_dir}")
            return []
        c3d_files = sorted(subject_dir.glob("*.c3d"))
        out = []
        meta = self.able_bodied_metadata.get(subject_id.upper(), {})
        
        for trial_idx, c3d_file in enumerate(c3d_files):
            mocap = self.load_c3d_file(c3d_file)
            if mocap is None:
                continue
            trial_raw = c3d_file.stem
            trial_name = sanitize(trial_raw)
            out.append({
                **meta, **mocap,
                "trial_index": trial_idx,
                "trial_name": trial_name,
                "trial_name_raw": trial_raw,
                "dataset": "van_criekinge",
                "data_type": "mocap"
            })
        return out

    def process_stroke_subject(self, subject_id: str) -> List[Dict]:
        if subject_id.startswith("TVC"):
            subject_dir = self.base_data_dir / "stroke" / subject_id
        else:
            subject_dir = self.base_data_dir / "stroke" / subject_id
        if not subject_dir.exists():
            logger.warning(f"Missing stroke subject dir: {subject_dir}")
            return []
        c3d_files = sorted(subject_dir.glob("*.c3d"))
        out = []
        meta = self.stroke_metadata.get(subject_id.upper(), {})
        for trial_idx, c3d_file in enumerate(c3d_files):
            mocap = self.load_c3d_file(c3d_file)
            if mocap is None:
                continue
            trial_raw = c3d_file.stem
            trial_name = sanitize(trial_raw)
            out.append({
                **meta, **mocap,
                "trial_index": trial_idx,
                "trial_name": trial_name,
                "trial_name_raw": trial_raw,
                "dataset": "van_criekinge",
                "data_type": "mocap_stroke"
            })
        return out

    # ------------ marker layout ------------
    def create_marker_layout_config(self, sample_trial_data: Dict) -> Dict:
        names_original = sample_trial_data['marker_names']

        cleaned_names = []
        for m in names_original:
            m_clean = m.strip().upper()
            if ":" in m_clean:
                m_clean = m_clean.split(":")[-1]
            cleaned_names.append(m_clean)
        layout = {
            "marker_set_name": "van_criekinge_full",
            "total_markers": len(cleaned_names),
            "marker_names": cleaned_names,
            "description": "Van Criekinge gait lab marker set with 3D positions and computed angles",
            "body_parts": {k: [] for k in ["head","torso","left_arm","right_arm","pelvis","left_leg","right_leg","computed_angles"]},
            "marker_types": {"position_markers": [], "angle_markers": []},
        }

        for i, u in enumerate(cleaned_names):
            if u in ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'HEDO', 'HEDA', 'HEDL', 'HEDP']:
                layout['body_parts']['head'].append(i); layout['marker_types']['position_markers'].append(i)
            elif u in ['C7', 'T10', 'CLAV', 'STRN', 'TRXO', 'TRXA', 'TRXL', 'TRXP']:
                layout['body_parts']['torso'].append(i); layout['marker_types']['position_markers'].append(i)
            elif u in ['LASI', 'RASI', 'SACR', 'PELO', 'PELA', 'PELL', 'PELP']:
                layout['body_parts']['pelvis'].append(i); layout['marker_types']['position_markers'].append(i)
            elif u in ['LSHO', 'LELB', 'LWRA', 'LWRB', 'LFIN'] or (u.startswith('L') and any(x in u for x in ['HUO','HUA','HUL','HUP','RAO','RAA','RAL','RAP','HNO','HNA','HNL','HNP'])):
                layout['body_parts']['left_arm'].append(i); layout['marker_types']['position_markers'].append(i)
            elif u in ['RSHO', 'RELB', 'RWRA', 'RWRB', 'RFIN'] or (u.startswith('R') and any(x in u for x in ['HUO','HUA','HUL','HUP','RAO','RAA','RAL','RAP','HNO','HNA','HNL','HNP'])):
                layout['body_parts']['right_arm'].append(i); layout['marker_types']['position_markers'].append(i)
            elif u in ['LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE'] or (u.startswith('L') and any(x in u for x in ['FEO','FEA','FEL','FEP','TIO','TIA','TIL','TIP','FOO','FOA','FOL','FOP','TOO','TOA','TOL','TOP','CLO','CLA','CLL','CLP'])):
                layout['body_parts']['left_leg'].append(i); layout['marker_types']['position_markers'].append(i)
            elif u in ['RTHI', 'RKNE', 'RTIB', 'RANK', 'RHEE', 'RTOE'] or (u.startswith('R') and any(x in u for x in ['FEO','FEA','FEL','FEP','TIO','TIA','TIL','TIP','FOO','FOA','FOL','FOP','TOO','TOA','TOL','TOP','CLO','CLA','CLL','CLP'])):
                layout['body_parts']['right_leg'].append(i); layout['marker_types']['position_markers'].append(i)
            elif 'ANGLES' in u or u.startswith('*'):
                layout['body_parts']['computed_angles'].append(i); layout['marker_types']['angle_markers'].append(i)

        layout['body_part_counts'] = {k: len(v) for k, v in layout['body_parts'].items()}
        layout['key_markers'] = {
            'head': [i for i, n in enumerate(cleaned_names) if n in ['LFHD', 'RFHD', 'LBHD', 'RBHD']],
            'spine': [i for i, n in enumerate(cleaned_names) if n in ['C7', 'T10', 'SACR']],
            'pelvis': [i for i, n in enumerate(cleaned_names) if n in ['LASI', 'RASI', 'SACR']],
            'shoulders': [i for i, n in enumerate(cleaned_names) if n in ['LSHO', 'RSHO']],
            'elbows': [i for i, n in enumerate(cleaned_names) if n in ['LELB', 'RELB']],
            'wrists': [i for i, n in enumerate(cleaned_names) if n in ['LWRA', 'LWRB', 'RWRA', 'RWRB']],
            'knees': [i for i, n in enumerate(cleaned_names) if n in ['LKNE', 'RKNE']],
            'ankles': [i for i, n in enumerate(cleaned_names) if n in ['LANK', 'RANK']],
            'feet': [i for i, n in enumerate(cleaned_names) if n in ['LHEE', 'LTOE', 'RHEE', 'RTOE']],
        }
        layout['mosh_recommended_markers'] = layout['marker_types']['position_markers']
        return layout

    # ------------ config for potential downstream use (kept minimal) ------------
    def create_mosh_config(self, trial_data: Dict) -> Dict:
        return {
            'subject_id': trial_data['subject_id'],
            'gender': trial_data['gender'],
            'population': trial_data['population'],
            'height': trial_data['height_m'],
            'body_mass': trial_data['body_mass_kg'],
            'model_type': 'smpl',
            'frame_rate': trial_data['frame_rate'],
            'n_frames': trial_data['n_frames'],
            'conditioning': {
                'age': trial_data['age'],
                'sex': trial_data['sex'],
                'height': trial_data['height_m'],
                'mass': trial_data['body_mass_kg'],
                'condition': trial_data['condition'],
                'walking_speed': self._get_walking_speed(trial_data),
                'population': trial_data['population']
            }
        }

    def _get_walking_speed(self, trial_data: Dict) -> Optional[float]:
        if trial_data['population'] == 'able_bodied':
            s = trial_data.get('walking_speeds', {})
            l, r = s.get('left_speed'), s.get('right_speed')
            if l is not None and r is not None:
                return (l + r) / 2.0
        else:
            s = trial_data.get('walking_speeds', {})
            p, n = s.get('paretic_speed'), s.get('non_paretic_speed')
            if p is not None and n is not None:
                return (p + n) / 2.0
        return None

    # ------------ save ------------
    def save_processed_data(self, processed_trials: List[Dict], subject_id: str):
        subject_out = self.output_dir / subject_id
        subject_out.mkdir(parents=True, exist_ok=True)

        for trial in processed_trials:
            trial_name = trial['trial_name']
            # 1) marker layout
            layout = self.create_marker_layout_config(trial)

            # 2) split to position markers only
            idx_pos = layout['marker_types']['position_markers']
            data_full = trial['marker_data']              # (T, N_all, 3)
            data_pos = data_full[:, idx_pos, :]           # (T, K, 3)
            names_pos = [layout['marker_names'][i] for i in idx_pos]

            # 3a) save full (post basic frame filtering)
            np.savez(subject_out / f"{trial_name}_markers_full.npz",
                     marker_data=data_full,
                     marker_names=layout['marker_names'],
                     frame_rate=trial['frame_rate'],
                     marker_layout=layout)

            # 3b) gap fill (optional)
            if self.gap_fill_max > 0:
                T, K, _ = data_pos.shape
                for k in range(K):
                    for d in range(3):
                        data_pos[:, k, d] = _linear_fill_gaps(data_pos[:, k, d], self.gap_fill_max)

            # 3c) resample (optional)
            out_rate = trial['frame_rate']
            if self.resample_fps is not None and trial['frame_rate'] != self.resample_fps:
                if self.resample_mode == "decimate":
                    step = max(1, int(round(trial['frame_rate'] / self.resample_fps)))
                    data_pos = data_pos[::step]
                    out_rate = trial['frame_rate'] / step
                else:  # "interp"
                    T = data_pos.shape[0]
                    t_src = np.arange(T) / trial['frame_rate']
                    t_end = (T - 1) / trial['frame_rate']
                    n_tgt = int(round(t_end * self.resample_fps)) + 1
                    t_tgt = np.arange(n_tgt) / self.resample_fps
                    T_new, K = n_tgt, data_pos.shape[1]
                    data_new = np.full((T_new, K, 3), np.nan, dtype=float)
                    for k in range(K):
                        for d in range(3):
                            data_new[:, k, d] = _resample_series(data_pos[:, k, d], t_src, t_tgt)
                    data_pos = data_new
                    out_rate = float(self.resample_fps)

            # 4) save positions-only (clean)
            np.savez(subject_out / f"{trial_name}_markers_positions.npz",
                     marker_data=data_pos,
                     marker_names=names_pos,
                     frame_rate=out_rate,
                     marker_layout=layout,
                     original_indices=np.asarray(idx_pos, dtype=int))

            # 5) config + metadata
            cfg = self.create_mosh_config(trial)
            cfg['marker_data_shape'] = data_pos.shape
            cfg['position_markers_only'] = True
            cfg['n_position_markers'] = len(names_pos)
            cfg['marker_layout'] = layout
            with open(subject_out / f"{trial_name}_mosh_config.json", "w") as f:
                json.dump(cfg, f, indent=2)

            meta = {k: v for k, v in trial.items() if k not in ['marker_data']}
            
            # Added this line
            # Add cleaned marker names to metadata
            meta['marker_names'] = layout['marker_names']

            meta['marker_layout'] = layout
            meta['n_position_markers'] = len(names_pos)
            meta['n_angle_markers'] = len(layout['marker_types']['angle_markers'])
            with open(subject_out / f"{trial_name}_metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(f"[{subject_id}] saved {trial_name} | pos_markers={len(names_pos)} | fps={out_rate:.2f}")

    # ------------ dataset driver ------------
    def process_dataset(self, subject_subset: Optional[List[str]] = None):
        able_dir = self.base_data_dir / "able_bodied"
        on_disk_able = {
            p.name.upper() for p in able_dir.iterdir()
            if p.is_dir() and p.name.upper().startswith("SUBJ")
        }
        able_bodied_subjects = sorted(on_disk_able & set(self.able_bodied_metadata.keys()))

        if subject_subset:
            subject_subset = {s.upper() for s in subject_subset}
            able_bodied_subjects = [s for s in able_bodied_subjects if s in subject_subset]

        logger.info(f"Processing {len(able_bodied_subjects)} able-bodied subjects...")


        for sid in able_bodied_subjects:
            try:
                trials = self.process_able_bodied_subject(sid)
                if trials:
                    self.save_processed_data(trials, sid)
                    logger.info(f"Completed {sid}: {len(trials)} trials")
                else:
                    logger.warning(f"No valid trials for {sid}")
            except Exception as e:
                logger.error(f"Error on {sid}: {e}")

        # Stroke pass is optional; uncomment when ready
        # stroke = list(self.stroke_metadata.keys())
        # if subject_subset:
        #     stroke = [s for s in stroke if s in subject_subset]
        # logger.info(f"Stroke subjects: {len(stroke)}")
        # for sid in stroke:
        #     try:
        #         trials = self.process_stroke_subject(sid)
        #         if trials:
        #             self.save_processed_data(trials, sid)
        #             logger.info(f"Completed {sid}: {len(trials)} trials")
        #         else:
        #             logger.warning(f"No valid trials for {sid}")
        #     except Exception as e:
        #         logger.error(f"Error on {sid}: {e}")

        self.generate_dataset_summary()
        logger.info("Dataset processing completed.")

    # ------------ summary (unchanged logic) ------------
    def generate_dataset_summary(self):
        summary = {
            'dataset_name': 'van_criekinge_mocap',
            'total_subjects': 0,
            'total_trials': 0,
            'populations': {},
            'age_stats': {'min_age': None, 'max_age': None, 'mean_age': None,
                          'age_ranges': {'20-30': 0,'31-40': 0,'41-50': 0,'51-60': 0,'61-70': 0,'71-80': 0,'80+': 0}},
            'gender_distribution': {'male': 0, 'female': 0},
            'height_stats': {'min_height_m': None, 'max_height_m': None, 'mean_height_m': None},
            'mass_stats': {'min_mass_kg': None, 'max_mass_kg': None, 'mean_mass_kg': None},
            'walking_speed_stats': {'min_speed': None, 'max_speed': None, 'mean_speed': None},
            'marker_info': {'total_markers_per_trial': None, 'position_markers_per_trial': None, 'angle_markers_per_trial': None},
            'conditioning_features': ['age','sex','height','mass','condition','walking_speed','population']
        }

        ages, heights, masses, speeds_all = [], [], [], []
        for subj_dir in self.output_dir.iterdir():
            if not subj_dir.is_dir():
                continue
            summary['total_subjects'] += 1
            metas = sorted(subj_dir.glob("*_metadata.json"))
            summary['total_trials'] += len(metas)
            if not metas:
                continue
            with open(metas[0], "r") as f:
                meta = json.load(f)

            pop = meta.get('population', 'unknown')
            summary['populations'][pop] = summary['populations'].get(pop, 0) + 1
            g = meta.get('gender', 'unknown')
            if g in ['male', 'female']:
                summary['gender_distribution'][g] += 1

            a = meta.get('age'); h = meta.get('height_m'); m = meta.get('body_mass_kg')
            if a is not None:
                ages.append(a)
                bins = [('20-30', 30), ('31-40', 40), ('41-50', 50), ('51-60', 60),
                        ('61-70', 70), ('71-80', 80)]
                placed = False
                for k, ub in bins:
                    if a <= ub:
                        summary['age_stats']['age_ranges'][k] += 1; placed = True; break
                if not placed:
                    summary['age_stats']['age_ranges']['80+'] += 1
            if h is not None: heights.append(h)
            if m is not None: masses.append(m)

            ws = meta.get('walking_speeds', {})
            vals = [v for v in ws.values() if v is not None]
            if vals:
                speeds_all.append(sum(vals)/len(vals))

            if 'n_position_markers' in meta:
                summary['marker_info']['position_markers_per_trial'] = meta['n_position_markers']
            if 'n_angle_markers' in meta:
                summary['marker_info']['angle_markers_per_trial'] = meta['n_angle_markers']
                summary['marker_info']['total_markers_per_trial'] = meta['n_position_markers'] + meta['n_angle_markers']

        if ages:
            summary['age_stats'].update(dict(min_age=min(ages), max_age=max(ages),
                                             mean_age=round(sum(ages)/len(ages), 1)))
        if heights:
            summary['height_stats'].update(dict(min_height_m=round(min(heights),3),
                                                max_height_m=round(max(heights),3),
                                                mean_height_m=round(sum(heights)/len(heights),3)))
        if masses:
            summary['mass_stats'].update(dict(min_mass_kg=min(masses),
                                              max_mass_kg=max(masses),
                                              mean_mass_kg=round(sum(masses)/len(masses),1)))
        if speeds_all:
            summary['walking_speed_stats'].update(dict(min_speed=round(min(speeds_all),3),
                                                       max_speed=round(max(speeds_all),3),
                                                       mean_speed=round(sum(speeds_all)/len(speeds_all),3)))

        with open(self.output_dir / "dataset_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary: subjects={summary['total_subjects']} trials={summary['total_trials']}")

# ------------ CLI ------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Process Van Criekinge C3D to SMPL-ready markers")
    p.add_argument('--data_dir', required=True, type=str,
                   help='Path to raw van_criekinge data (download from Figshare - see DATA.MD)')
    p.add_argument('--output_dir', required=True, type=str,
                   help='Path to save processed data (typically data/processed_markers_all_2/)')
    p.add_argument('--subjects', nargs='*', default=None, help='Subjects to process (e.g., SUBJ01 SUBJ02)')
    p.add_argument('--test_run', action='store_true', help='Process first 3 subjects only')
    p.add_argument('--min_valid_ratio', type=float, default=0.5, help='Frame keep threshold')
    p.add_argument('--gap_fill_max', type=int, default=10, help='Max consecutive NaNs to fill (frames)')
    p.add_argument('--resample_fps', type=float, default=20.0, help='Target FPS for positions (use 0 to disable)')
    p.add_argument('--resample_mode', type=str, default='interp', choices=['interp','decimate'],
                   help='Resampling method (interp=linear time interpolation)')
    args = p.parse_args()

    subjects_to_process = None
    if args.test_run:
        subjects_to_process = ["SUBJ01", "SUBJ02", "SUBJ03"]
        logger.info("Test run: first 3 able-bodied subjects")
    elif args.subjects:
        subjects_to_process = [s.upper() for s in args.subjects]

    resample = None if (args.resample_fps is not None and args.resample_fps <= 0) else args.resample_fps
    pre = VanCriekingeDatasetPreprocessor(
        base_data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_valid_ratio=args.min_valid_ratio,
        gap_fill_max=args.gap_fill_max,
        resample_fps=resample,
        resample_mode=args.resample_mode
    )
    pre.process_dataset(subject_subset=subjects_to_process)

if __name__ == "__main__":
    main()
