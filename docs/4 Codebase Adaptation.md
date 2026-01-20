>[!info|right]
> **Part of a series on**
> ###### [[Dataset Overview|Van Criekinge]]
> ---
> **Overview**
> * [[Dataset Overview|Main Article]]
> * [[4 Codebase Adaptation|Codebase Adaptation]]
> * [[Known Bad Files|Known Issues]]

>[!info]
> ###### Codebase Adaptation
> [Type::Integration Guide]
> [Codebase::LoRA-MDM]
> [Feature::Age Conditioning]
> [Status::Implemented]

This document details the **technical integration** of the Van Criekinge dataset and the implementation of **age conditioning** within the LoRA-MDM codebase, based on work completed in December 2025.


## 1. Architecture Modification: Age Conditioning
To enable age-conditioned motion generation, the MDM architecture was modified to accept and process a scalar age parameter.

### AgeEncoder
A new `AgeEncoder` class was added to `model/mdm.py` to project the scalar age into the model's latent dimension.

```python
class AgeEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
    def forward(self, age):
        # age: [batch_size, 1]
        return self.fc(age)
```

### Integration in MDM Forward Pass
The age embedding is concatenated with timestep and text embeddings. 
- For `trans_enc`, it is concatenated to the input sequence.
- For `trans_dec`, it is added to the memory sequence for cross-attention.

## 2. Dataset Implementation: `AgeMotionDataset`
A dedicated dataset class was created in `data_loaders/age/dataset.py` to handle the VC dataset structure, which includes motion, text, and age labels.

### Key Logic
*   **Age Normalization**: Age values (e.g., 0-100) are normalized to [0, 1].
*   **Explicit Padding**: Motions shorter than the maximum length are explicitly zero-padded to maintain consistency.
*   **Variable Shadowing Fix**: Updated loops to use `frame_idx` to avoid shadowing the original `idx`.

```python
# data_loaders/age/dataset.py snippet
def __getitem__(self, item):
    # ...
    frame_idx = random.randint(0, len(motion) - m_length)
    motion = motion[frame_idx:frame_idx+m_length]
    
    # Explicit Padding
    if m_length < self.max_motion_length:
        motion = np.concatenate([
            motion,
            np.zeros((self.max_motion_length - m_length, motion.shape[1]))
        ], axis=0)
    # ...
```

## 3. Bug Fixes & Integration Lessons

Several critical bugs were identified and fixed during the integration process:

1.  **`trans_dec` Memory Path**: Initially, the age embedding was calculated but ignored in the default `trans_dec` path (where `emb_trans_dec=False`). The fix involved concatenating `age_emb` to the `memory` tensor before passing it to `seqTransDecoder`.
2.  **Prior Preservation Slicing**: In `training_loop.py`, the code failed to slice the `age` tensor when handling smaller end-of-epoch batches, leading to shape mismatches.
3.  **Dtype Mismatch**: Ensured `agebatch` is cast to `torch.float32` in `tensors.py` to avoid "mat1 and mat2 must have the same dtype" errors.
4.  **LoRA Weight Filtering**: Updated `only_lora` in `training_loop.py` to preserve `age` related weights during finetuning.

## 4. Validation via Velocity-to-Age Mapping

To verify the age conditioning before using ground truth labels, a validation test was performed using a "synthetic" age based on average root velocity.

- **Hypothesis**: High velocity → Young (Low Age), Low velocity → Old (High Age).
- **Result**: After training LoRA layers on this mapping, the model showed a **-0.9691 correlation** between generated age and resulting motion velocity.
- **Conclusion**: Success! The age scalar effectively influences the generated motion dynamics.

> [!NOTE]
> For more details on the validation script, see `age_velocity_mapping.py`.
