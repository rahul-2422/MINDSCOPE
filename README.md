# MINDSCOPE: Machine-learning Inferencing of NeuroData for Seamless Cognitive Overload Prediction and Evaluation

---

**MINDSCOPE is a comprehensive project that explores the use of machine learning techniques to analyze and classify brain signals. This repository contains the code and resources required to run the various stages of the MINDSCOPE project.**

## Prerequisites
Before running the code, ensure that you have the following setup:

* Clone the MINDSCOPE repository to your local machine.
* Create the following directories within the project folder:
  1. preprocessed-data
  2. channel-selection-data
  3. features-data
* Create a Python virtual environment (recommended) and activate it:
   `python -m venv env`
    `source env/bin/activate` (on Windows, use `env\Scripts\activate`)
* Install the required packages using the provided requirements.txt file:
    `pip install -r requirements.txt`

*The use of a virtual environment is recommended, as it helps to avoid potential conflicts between system packages and the packages required for this project.*

## Running the Code
MINDSCOPE consists of the following stages, which should be executed in the given order:

  1.  preprocessing.py
  2.  channel-selection.py
  3.  feature-extraction.py
  4.  classification.py
  5.  model-trainer.py

Each script performs a specific task, and the output of one script is often used as the input for the next. Make sure to run the scripts in the order listed above.

## Citations
The following resources were helpful in the completion of the MINDSCOPE project:

* [Nolan, Hugh, Robert Whelan, and Richard B. Reilly. "FASTER: fully automated statistical thresholding for EEG artifact rejection." Journal of neuroscience methods 192.1 (2010): 152-162.](https://www.sciencedirect.com/science/article/pii/S0165027010003894)
* [Lan, Tian, et al. "Channel selection and feature projection for cognitive load estimation using ambulatory EEG." Computational intelligence and neuroscience 2007 (2007).](https://www.hindawi.com/journals/cin/2007/074895/abs/)
