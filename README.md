# Data Science King County Home Estimator: Project Overview
* A tool to estimate the home prices (MAE ~ 75,000) in King County, WA to help home buyers negotiate  
* Dataset of 21,613 King County homes sold between May 2014 - May 2015 obtained from Kaggle
* Looked into many different approaches including Linear, Lasso, Random Forest and XGBoost regressors to find best MAE
* Optimized Random Forest and XGBoost regressors to reach the optimal model with an accuracy of 96%
* Built and integrated model into restful API using Flask

## Code and Resource Reference
**Python Version:** 3.7
**Packages:** Pandas, Numpy, Sklearn, Matplotlib, Seaborn, Selenium, Flask, Json, Pickle
**Web Framework Requirements:** ```pip install -r requirements.txt```
**Kaggle Dataset:** https://www.kaggle.com/harlfoxem/housesalesprediction
**Flask Productionization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## Project Walkthough
This is my second data science project following the same youtube walkthrough. Huge shout out to Ken Jee for simplifying the data science process as well as providing a vast amount of insight on the projects Data Scientists work on: https://www.youtube.com/playlist?list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

## Cleaning the Data
Dataset taken from Kaggle did not need much cleaing. Only removed trailing zeros from some columns to increase legibility.

## EDA
Examining distributions of the data, it was interesting to see the correlations between home attributes and price. Here are some hightlights.

![](EDA%20images/corr2.jpg)

## Building the Models

## Performance of Models

# Productionization using Flask



