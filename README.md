# Self-Attention-Based Contextual Modulation Improves Neural System Identification

This repository serves as supporting material for the paper titled Self-Attention-Based Contextual Modulation Improves Neural System Identification.
The README is structured as follows:

1. Dataset Sample
2. Modeling
3. Traditional (Simultaneous) Training
4. Incremental Training
5. Tuning Curves
6. Calculating Neuronal Tuning Metrics: Correlation (CORR.) and Peak Tuning (PT)
7. FCL Decomposition
8. Attention Highlighting

## 1. Dataset Sample

<div align="center">
  <img src="./assets/data_info.png" width=600"/>
</div>


As the dataset has not been publicly released, we provide a small subsample of the monkey 1 site 1 (M1S1) data. The sample contains the 

* Training images
* Training responses
* Validation images
* Validation responses

See the analysis/data_sample.ipynb jupyter notebook for more details.

## 2. Modeling
<div align="center">
  <p float="center">
    <img src="./assets/main_models.png" width="500" />
    <img src="./assets/inc_models_hq.png" width="250" /> 
  </p>
</div>



Models can be found in the modeling folder.

## 3. Traditional (Simultaneous) Training

Traditional training scripts can be found in training folder.

## 4. Incremental Training

Incremental training scripts can be found in training folder. Note the incremental loading of parameters off of already trained models.

## 5. Tuning Curves


<div align="center">
  <img src="./assets/tuning_curves.png" width="600"/>
</div>

<div align="center">
  <img src="./assets/avg_tc.png" width="600"/>
</div>


Generation of tuning curves found in analysis. See individual neuron tuning curves, and population average tuning curves.

## 6. Calculating Neuronal Tuning Metrics: CORR. and PT

See analysis folder. CORR. via pearson correlation function. PT_J and PT_S based on ranking and top 10 (1%).

## 7. FCL Decomposition

<div align="center">
  <img src="./assets/fcl_decomp.png" width="600"/>
</div>

Extracting contributions found in analysis. Requires modifying model to save extra parameters.

## 8. Attention Highlighting

<div align="center">
  <img src="./assets/att_hlight.png" width="600"/>
</div>

Extracting HH map, and then querying center. Requires modifying model to save extra parameters.















