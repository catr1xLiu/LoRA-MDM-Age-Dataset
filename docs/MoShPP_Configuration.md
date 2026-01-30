# MoSh++ Body Model Configuration

## Model Selection Method

- MoSh++ selects the body model primarily from the OmegaConf configuration key `surface_model.type` (default is `smplx`). See `ExternLib/moshpp/support_data/conf/moshpp_conf.yaml` where `surface_model.type: smplx` and `surface_model.fname` is derived from that type.
- The actual model file is given by `surface_model.fname` (default: `${dirs.support_base_dir}/${surface_model.type}/${surface_model.gender}/model.pkl`). The code asserts the file exists early in `MoSh.__init__` (`ExternLib/moshpp/src/moshpp/mosh_head.py`).
- When loading the `.pkl` model, the loader (`load_surface_model` in `ExternLib/moshpp/src/moshpp/models/smpl_fast_derivatives.py`) will also infer model type from the pickled data if an explicit `surface_model_type` is not supplied (it inspects `posedirs` shape and maps it to types). However many code paths (marker layout generation, optimizer behavior, priors, pose splitting) depend on `cfg.surface_model.type`, so you should set `surface_model.type` to the correct value rather than relying only on the file contents.

Files to inspect (reference): `ExternLib/moshpp/src/moshpp/mosh_head.py`, `ExternLib/moshpp/src/moshpp/models/bodymodel_loader.py`, `ExternLib/moshpp/src/moshpp/models/smpl_fast_derivatives.py`, `ExternLib/moshpp/support_data/conf/moshpp_conf.yaml`.

## Directory Structure

# For SMPL-X:
data/soma_support/smplx/[gender]/model.pkl

# For SMPL:
data/soma_support/smpl/[gender]/model.pkl

# For SMPL+H (SMPLH):
data/soma_support/smplh/[gender]/model.pkl

Notes:
- The default `surface_model.fname` template is set in `moshpp_conf.yaml`:
  - `surface_model.fname: ${dirs.support_base_dir}/${surface_model.type}/${surface_model.gender}/model.pkl`
- Supporting files (priors, dmpls) are expected under `dirs.support_base_dir/${surface_model.type}/` (see next section).

## Configuration Parameters (exact names)

- `surface_model.type` (string) — model type used across code (e.g. `smpl`, `smplh`, `smplx`, `mano`, `object`, `animal_horse`, `animal_dog`). Default in config: `smplx`.
- `surface_model.fname` (string) — path to the model `.pkl` file; default derived from `dirs.support_base_dir` + `surface_model.type` + `surface_model.gender`.
- `surface_model.gender` (string) — usually `male`/`female`/`neutral`; used to form `surface_model.fname`.
- `surface_model.dmpl_fname` (string) — path to `dmpl.pkl` (dynamics PCA) for dynamic models.
- `surface_model.num_betas`, `surface_model.dof_per_hand`, `surface_model.num_expressions` — numeric settings used by loader/fitting.
- `moshpp.pose_body_prior_fname` — path to `pose_body_prior.pkl` (usually `${dirs.support_base_dir}/${surface_model.type}/pose_body_prior.pkl`).
- `moshpp.pose_hand_prior_fname` — path to `pose_hand_prior.npz` (usually `${dirs.support_base_dir}/${surface_model.type}/pose_hand_prior.npz`) — required for `smplx`/`smplh`/`mano`.
- `moshpp.v_template_fname`, `moshpp.betas_fname` — optional overrides for template and betas.

How to override at runtime:
- Pass a config dict or dotlist overrides to `MoSh` / `MoSh.prepare_cfg`. Example:

```python
from moshpp.mosh_head import MoSh
# override via kwargs/dotlist (this is what prepare_cfg accepts)
mp = MoSh(dict_cfg={'surface_model':{'type':'smpl'}})
# or using keyword override in constructor
mp = MoSh(**{'surface_model.type':'smpl'})
```

Important implementation detail: while `surface_model.fname` (the `.pkl`) can be used to infer the model shape, many branches in the codebase branch on `cfg.surface_model.type`. So set `surface_model.type` to the intended model (preferred) and point `surface_model.fname` to the corresponding `model.pkl`.

## Model File Requirements (per model type)

- Common requirement for SMPL / SMPLH / SMPLX / MANO: a pickled `.pkl` model file that contains the model dict expected by `load_surface_model`:
  - keys used: `posedirs`, `shapedirs`, `v_template` (optional), `kintree_table`, `bs_style` (should be `'lbs'`), `weights`, `J_regressor`, `f` (faces), plus other arrays (the loader inspects `posedirs.shape[2]` to infer pose-parameter counts).
  - The loader asserts the filename ends with `.pkl`.

- SMPL (type `smpl`):
  - `model.pkl` in `.../smpl/[gender]/model.pkl`.
  - Optional: `pose_body_prior.pkl` in `.../smpl/pose_body_prior.pkl` if you want body pose priors.
  - `pose_hand_prior.npz` is NOT required for plain `smpl` (hands handled differently than `smplh`/`smplx`).

- SMPL+H (type `smplh`):
  - `model.pkl` in `.../smplh/[gender]/model.pkl`.
  - Required/expected: `pose_hand_prior.npz` in `.../smplh/pose_hand_prior.npz` (loader expects a hand prior for `smplh`).
  - `pose_body_prior.pkl` in `.../smplh/pose_body_prior.pkl` is commonly present.

- SMPL-X (type `smplx`):
  - `model.pkl` in `.../smplx/[gender]/model.pkl` (your current error shows `data/soma_support/smplx/neutral/model.pkl`).
  - Required: `pose_hand_prior.npz` in `.../smplx/pose_hand_prior.npz` and often `pose_body_prior.pkl` in `.../smplx/pose_body_prior.pkl`.
  - For dynamics fitting `dmpl.pkl` may be present at `.../smplx/[gender]/dmpl.pkl` and referenced by `surface_model.dmpl_fname`.

- MANO (type `mano`):
  - MANO `.pkl` (e.g. `MANO_RIGHT.pkl`) — loader treats `mano` specially (different pose partitioning). Hand prior `.npz` is used.

- Object models (type `object`):
  - `RigidObjectModel` expects a mesh `ply` file — bodymodel loader treats `object` differently (see `bodymodel_loader.load_moshpp_models`).

Additional notes:
- Marker-layout files are named `${surface_model.type}_${mocap.ds_name}.json` by default (see `mosh_head.py` line that sets `dirs.marker_layout.fname`). Marker layout code also expects `marker_meta['surface_model_type']` to match `cfg.surface_model.type`.
- The loader function `load_surface_model` contains a map of joint-parameter counts to types (69->smpl, 153->smplh, 162->smplx, 45->mano, 105->animal_horse, 102->animal_dog). If you provide a `.pkl` that matches one of those shapes, the loader can deduce the type, but again many places read `cfg.surface_model.type` so keep them consistent.

If you want, I can: (1) show the exact code snippet you need to change in your project to switch to `smpl`, or (2) run a quick check to list `data/soma_support/*` to verify which model folders/files you already have. Reply with 1 or 2.
