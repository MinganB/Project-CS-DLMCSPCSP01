# Project CS DLMCSPCSP01
 IUBH MSc Computer Science Project DLMCSPCSP01.

 ## Model background
An adaptation of the standard SIR epidemiological model will be applied to attempt to model the general patterns of misinformation spread. Parameters for the adapted model will then be estimated from a real-world misinformation dataset and the model outcomes compared to the real-world data values to estimate the accuracy of the model.

## Model / application usage
### Getting started
1. The Python virtual environment can be entered by running 'source bin/activate' in the 'sir_venv/Project-CS-DLMCSPCSP01' working directory.
2. The project dependancies can be installed from 'requirements.txt' by running 'python -m pip install -r requirements.txt'

### Data analysis
1. The Covid-19 misinformation dataset should be placed in 'sir_venv/Project-CS-DLMCSPCSP01/data/covid-misinfo-videos.csv'. Once data analysis is complete the output will be located in 'sir_venv/Project-CS-DLMCSPCSP01/data/covid-misinfo-videos-cleaned.csv'.
2. The data cleaning and analysis can be performed by running the data analysis module: 'python data_analysis.py'. This will output the cleaned dataset and presentation graphs, as well as printing the derived model parameters to the console. These parameters can be used to inform the model.

### SIR model
