# UNI-IQA
A pytorch implementation of the paper ''UNI-IQA: A Unified Approach for Mutual Promotion of Natural and Screen Content Image Quality Assessment"

![image](https://github.com/democode123/UNI-IQA/blob/main/pipeline.png)

# Prequisite:
Python 3+  
PyTorch 1.4+  
Matlab  

# Usage
## Sampling image pairs from multiple databases
data_all.m  
## Combining the sampled pairs to form the training set
combine_train.m  
## Training on multiple databases for 10 sessions
python main.py
## Result anlysis
Compute SRCC/PLCC after nonlinear mapping: result_analysis.m  
Compute fidelity loss: eval_fidelity.m

# Links to download the NI and SCI datasets used in the experiment
NI datasets: <a href="https://github.com/zwx8981/UNIQUE">Baidu Netdisk</a> (code:)   SCI datasets: <a href="https://github.com/zwx8981/UNIQUE">Baidu Netdisk</a>(code:)

# Notice
Our code is base on <a href="https://github.com/zwx8981/UNIQUE">UNIQUE</a>, we are truly grateful
for the authors' contribution.
