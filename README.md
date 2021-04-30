# DeepSHAP
Repository for the DeepSHAP experiments.

## Prerequisites

* Python, NumPy, Tensorflow, Keras, XGBoost.

## Experiments

Experiments for evaluating baseline distributions are in:

* `1_multiple_references/`

Experiments for evaluating series of models are in:

* `2_gene_expression_pathway/`
* `3_loss_explanation/`
* `4_feature_extraction/`
* `5_model_stack/`

## Code

Code underlying the experiments and implementations of DeepSHAP for our specific applications is found in `deepshap/`.

## Dataset availability

The NHANES I, NHANES 1999-2014, CIFAR, and MNIST data sets are all publicly available.  The HELOC data set can be obtained by accepting the data set usage license: (https://community.fico.com/s/explainable-machine-learning-challenge?tabset-3158a=a4c37).  Metabric data access is restricted and requires getting an approval through Sage Bionetworks Synapse website: https://www.synapse.org/#!Synapse:syn1688369 and https://www.synapse.org/#!Synapse:syn1688370.  ROSMAP data access is restricted and requires getting an approval through Sage Bionetworks Synapse website: https://www.synapse.org/#!Synapse:syn3219045 and is available as part of the AD Knowledge Portal https://adknowledgeportal.synapse.org/.
