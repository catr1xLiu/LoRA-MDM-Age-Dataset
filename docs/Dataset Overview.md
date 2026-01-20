
>[!info|right]
> **Part of a series on**
> ###### [[Dataset Overview|Van Criekinge]]
> ---
> **Overview**
> * [[Dataset Overview|Main Article]]
> * [[4 Codebase Adaptation|Codebase Adaptation]]
> * [[Known Bad Files|Known Issues]]
> 
> **Modules**
> * [[1 VC Pipeline/Pipeline Overview|Pipeline]]
> * [[2 Closer Look at VC/1 Current Pipeline Failures|Analysis]]
> * [[3 Solutions/New Marker Fitting|Solutions]]
> * [[Tools/Visualization Setup|Tools]]

>[!info]
> ###### Van Criekinge Gait Dataset
> [Type::Public Dataset]
> [Subject::Gait Analysis]
> [Participants::188 Total]
> [Format::C3D / MATLAB]
> [Pipeline::Integrated]

>[!QUOTE] 
>The first half of any good model is the dataset.

The **Van Criekinge Gait Dataset** is a publicly available motion capture dataset containing full-body biomechanical data of human walking. Published in 2023 by researchers led by Tamaya Van Criekinge at the University of Antwerp and KU Leuven, it comprises recordings from **138 healthy adults** spanning ages 21 to 86 and **50 stroke survivors**. The dataset is notable for its multimodal capture (kinematics, kinetics, and electromyography), lifespan coverage, and inclusion of a clinical population, making it one of the most comprehensive open-access gait datasets available.

## Data Formats

The raw data is provided in 2 different formats.



|Format|Contents|
| --- | --- |
| **C3D** | Raw source files containing marker trajectories, analog data, and computed outputs|
| **MAT** | MATLAB files with stride-normalized kinematics, kinetics, and EMG |


## Pipeline & Integration

For the first part of the term, there was an attempt to fix then Van Criekinge C3D to HumanML3d format. See [[Pipeline Overview|Pipeline]].

Several problems were fixed, including name cleaning, as well as motion trajectories. See [[dataset_prep.py fixes]] and [[motion_process.py fixes]].

### Issues & Solutions

It was not until much later that a more fundamental problem of the dataset was discovered. See [[1 Current Pipeline Failures]].

To resolve this, an investigation of the original dataset were conducted in [[2 Explore VC C3D]] and [[3 Explore VC MATLAB]] as well as an investigation of the old fitting code in [[Old Fitting Code Explanation]].

Eventually a solution emerged in the form of [[New Marker Fitting]].

For integration of this dataset into LoRA-MDM-age, see [[4 Codebase Adaptation]]. Also visit [[Known Bad Files]] to gain insight on which file to remove from the dataset because of bad data.

