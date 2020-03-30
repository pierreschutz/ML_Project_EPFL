# Machine Learning EPFL 2019 - Project 1

## Title : Higgs Boson Machine Learning Challenge

### Abstract :

The discovery of the Higgs boson announced July 4, 2012 led to a number of prestigious awards, including a Nobel prize.
ATLAS, a particle physics experiment taking place at CERN, observed a signal of the Higgs boson decaying into two tau particles. 
Nevertheless, the signal is small and in a noisy background. This report details our work on the Higgs
boson Machine Learning Challenge, where the goal is to use machine learning methods to improve the
discovery significance of the ATLAS experiment. Using simulated data characterizing events detected by ATLAS,
the task was to classify events into "tau tau decay of a Higgs boson" vs "background".

### Introduction :

This project aims to make student learn the basic machine learning techniques to solve a problem into a competition format.
The students are given a train dataset and a test dataset and has to compute predictions and submit them to the
[aicrowd platfrom](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019).

The dataset contains 250,000 exemple that list 30 values calculated from a simulated ALTAS experiment
(more details about the features can be found [here](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf)).
The result we need to obtain is a prediction (1) or (-1) if we detect the decay of an Higgs boson (or not).
Some features values can be missing or impossible to compute and have been set to -999.

### Directory structure

The following directory contrains different text documents, code and data files. The structure is details below :

#### Documents

- [Project Description](project1_description.pdf): Describe the task to perform and the tools availables.
- [Challenge Documentation](documentation_v1.8.pdf): Describe the original challenge task and details the dataset features.
- [Project Report](project1_report.pdf): Describe our approach, work, and conclusion while solving the problem.

#### Code

##### Jupyter Notebooks
- [Model Exploration](./scripts/model_exploration.ipynb): Test different machine learning models on the data to select the most performant.
- [Parameters Selection](./scripts/params_selection.ipynb): Use cross validation on the most performant model to define the best data processing and regularisation parameter.
- [Final Model](./scripts/final_model.ipynb): Use the best model and parameters to achieve the best accuracy and create a submission.

##### Python scripts

- [Implementations.py](./scripts/implementations.py): All possible machine learning models available to use.
- [Run.py](./scripts/run.py): Optimal data preprocessing and model to solve the problem.


- [Process Data](./scripts/process_data.py): All the preprocessing methods and tools used.
- [Proj1 Helpers](./scripts/proj1_helpers.py): General helpers methods for file submission, prediction, trianing and data splitting.
- [Least squares helpers](./scripts/least_squares_helpers.py): Helpers methods for least squares models (costs, optimisation).
- [Logistic regression helpers](./scripts/logistic_regression_helpers.py): Helpers methods for logistic regression (costs, optimisation).
