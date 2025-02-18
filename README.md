# Predicting_Failure_NASA_Turbofan_Jet_Engine
I perform a statistical analysis on engine monitoring system data to predict and prevent catastrophic failure of a NASA turbofan jet engine

The dataset from kaggle can be found here
```
https://www.kaggle.com/datasets/behrad3d/nasa-cmaps/data
```
More information on the dataset can be found here
```
https://ieee-dataport.org/documents/nasa-turbofan-jet-engine-data-set
```

# Goal
The goal of this project is to predict the Remaining Useful Life (RUL) of a NASA Turbofan jet engine i.e approximately how many cycles the machine can be used before failure, based on flight the flight history of several hundred engines run until failure, under different conditions.

# Analysis approach and Data Structure
The data provided is time series data from 26 sensors aboard several hundred jet engines when under normal use. There are many approaches to processing time series data in the field of to predict machine failure and RUL, some of which involve deep neural networks such as Recurrent Neural Nets, Convolutional Neural Nets, Transformers, etc. In this repository, I will explore the benefits of using a deep learning approach, but will begin by establishing a standard by training a Hidden Markov Model (HMM). 

Predicting failure using a HMM is a commonly used approach because it provides an more interpretable way for us to understand how a decision on Remaining Useful Life (RUL) is made by the machine. HMMs begin by defining hidden internal states of the system, based on a set of final observables (Failure of non failure), and using these internal states to predict a of the system health, and machine RUL. 

## Data structure
The data, understood by an exploratory analysis, is organized as a set of "space" separated text files with 26 columns describing the operating conditions and onboard sensor data of a collection of jet engines identified through the "unit" column. The column "unit" represents a single engine used for analysis, where there are 100 individual engines in the dataset. A given jet engines say with unit_num=1, has an associated characteristic called "cycle_num" which is the number of instances the engine was powered on and put through the normal stresses of a flight. The engines were repeatedly cycled on until failure.

# Training the HMM


# Training the RNN
