# Project CS DLMCSPCSP01: Modeling misinformation spread using an adapted SIR model
 IUBH MSc Computer Science Project DLMCSPCSP01.

 ## Table of Contents
1. [Model Background](#model-background)
2. [File Structure](#file-structure)
3. [Model / Application Usage](#model--application-usage)
   - [Getting Started](#getting-started)
   - [Data Analysis](#data-analysis)
   - [SIR Model](#sir-model)
4. [Application Overview](#application-overview)
   - [Data Analysis](#data-analysis-1)
   - [Differential Solver](#differential-solver)
   - [SIR Module (Main Application Module)](#sir-module-main-application-module)

 ## Model background
An adaptation of the standard SIR epidemiological model will be applied to attempt to model the general patterns of misinformation spread. Parameters for the adapted model will then be estimated from a real-world misinformation dataset and the model outcomes compared to the real-world data values to estimate the accuracy of the model.

## Model / application usage
### Getting started
1. Clone the GitHub repository to your local machine.
2. Create and enter the Python virtual environment by running the following commands in the 'sir_venv/Project-CS-DLMCSPCSP01' working directory:
- Create the venv: 'python -m venv sir_venv'
- Enter the venv: 'source bin/activate' 
4. The project dependancies can be installed from 'requirements.txt' by running:
- 'python -m pip install -r requirements.txt'

### Data analysis
The data analysis module may be used to extract parameters informing the SIR model.
1. The Covid-19 misinformation dataset should be placed in 'sir_venv/Project-CS-DLMCSPCSP01/data/covid-misinfo-videos.csv'. Once data analysis is complete the output will be located in 'sir_venv/Project-CS-DLMCSPCSP01/data/covid-misinfo-videos-cleaned.csv'.
2. The data cleaning and analysis can be performed by running the data analysis module: 'python data_analysis.py'. This will output the cleaned dataset and presentation graphs, as well as printing the derived model parameters to the console. These parameters can be used to inform the model.

### SIR model
Once data analysis has been completed, the SIR module can be used to create and fit model.
1. Parameter inputs can either be specified in the sir.py module (lines 100 - 109), or 'enable_gui' can be set to True (line 79) allowing input through a user interface.
2. Run the module: 
- 'python sir.py'

## Application overview
### Data analysis
The Python application includes a data analysis module to process and analyze the real-world misinformation dataset. The module is responsible for:
- Cleaning the dataset by removing rows with invalid or missing columns.
- Dropping redundant columns.
- Generating real-world visualization graphs depicting the number of user engagements with each video over time.
- Estimating daily engagements by calculating average daily rates.

### Differential solver
A differential solver module was developed to serve as a generic interface for differential equation solvers. Within this module, a Forward Euler solver was implemented and utilised in the main SIR module for finding derivatives.

### SIR module (main application module)
The parameters derived from the data analysis module can be input into the SIR module via the user interface to generate estimations and visualize the outcomes. The SIR module handles:
- The computation and graphical representation of the constructed model.
- Providing a comparison between the real-world data and the SIR predictions.
