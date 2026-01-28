# First Month Summary – Human Motion Model Project

This document summarizes the progress made during the first three weeks of the Human Motion Model research effort, focusing on key achievements, challenges encountered, and updated timelines.

---

## Overview
The project aims to generate realistic human motion conditioned on age groups using a diffusion‑based model (LoRA‑MDM). The initial months are dedicated to building foundational knowledge, preparing environments, evaluating datasets, and establishing a reproducible pipeline.

---

## Week 1 Highlights
- **Environment & Tool Setup** – Installed Obsidian, Typst, WSL, and set up remote GPU access (Beatrix). Prepared documentation repository structure on the NRC GitLab server.
- **Foundational Learning** – Completed self‑paced study of PyTorch fundamentals, transformer architecture, and basic diffusion models. Scheduled 90 min learning sessions daily.
- **Codebase Familiarisation** – Ran inference on MDM and LoRA‑MDM to confirm functionality. Generated a skeleton animation using the original code (Frame.jpg).
- **Project Assessment & Focus Shift** – Realised that continuous age conditioning via architecture changes is too ambitious given time constraints. Proposed shifting to discrete age groups, leveraging existing LoRA adapters, which offers a more realistic path to publishable results by mid‑February.

---

## Week 2 Highlights
- **Extended Learning** – Deepened understanding of PyTorch neural networks, diffusion model training pipelines, and the LoRA‑MDM paper. Trained a simple network for hands‑on practice.
- **Van Criekinge Dataset Analysis** – Developed scripts to visualize mesh quality; identified corrupted skeletons and inflated body shapes affecting the first 0.5 seconds of motion capture files.
- **Age Validation Approach** – Proposed using age recognition from generated motions as a validation metric, illustrated with an image (Drawing 1.16.svg).
- **Timeline Adjustments** – Updated project milestones: Week 3 will focus on dataset processing and training experiments; subsequent weeks allocate time for LoRA adapter training, abstract submission, fine‑tuning, and full paper writing.

---

## Week 3 Highlights
- **Pipeline Audit & Containerisation** – Replaced flawed Step 2→Step 3 conversion with a Dockerised MoSh++ + HumanML3D pipeline. Overcame Python 2 dependency issues by compiling from source on Ubuntu 20.04.
- **Dataset Splitting** – Parsed metadata to create balanced age groups (Young, Mid, Elderly) and produced distribution statistics (age vs. walking speed). Scripts will be integrated once motion data is re‑processed.
- **Documentation** – Completed an 8‑page technical analysis of pipeline failures and coordinate system errors, now available in the repository.
- **Timeline Update** – Confirmed completion of Weeks 1–3 tasks; Week 4 will focus on age‑based splitting, pipeline diagnosis, and a revised STGCN baseline training. Subsequent weeks follow the plan to train LoRA adapters, submit an abstract (Feb 15), and finish paper writing by Weeks 8‑9.

---

## Current Status & Next Steps
- **Environment** – Stable Docker image for MoSh++ + HumanML3D; remote GPU access granted.
- **Dataset** – Cleaned and split the Van Criekinge dataset into age groups. Ready for re‑processing of motion clips.
- **Model Development** – LoRA adapters will be trained in Weeks 5–7, followed by validation experiments.
- **Deliverables** – Targeting an extended abstract submission on Feb 15; full paper writing planned for Weeks 8‑9.

The project remains on a tight but achievable schedule. The shift to discrete age groups is expected to deliver publishable results within the set timeframe.
