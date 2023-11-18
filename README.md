# GRIDPredict
10-708F23 Project Code

# Folders

## Classes
contains reoccuring classes that are useful for data-processing

## Input
csv data

## Networks
contains classes and utilities for specific PGM methods


# Executables

## Data Processing

data_pipeline.py : generates cleaned data files


## Inference

parameter_learning_example.py : template used for (experiment.py)
experiment.py : trains a simple Bayesian Network to inference solar generation (used in midterm report)

## Structure Learning

structure_example.py : edit of PGMPY structure learning example that produces visualizations of the final graphs
BN_struc_learning.py : uses pgmpy functions to learn structure for Bayesian Network Inference
