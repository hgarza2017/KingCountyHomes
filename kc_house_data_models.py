
import pandas as pd
import numpy as np

df = pd.read_csv('kc_house_data_cleaned.csv')

df.columns

# choose columns to use
df_model = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'grade', 'sqft_above', 'sqft_basement',
               'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15']]


# train test split
from sklearn.model_selection import train_test_split

X = df_model.drop('price', axis = 1)
y = df_model.price.values
#ya = df_model.price this is the series which we shouldn't use

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# multiple linear regresssion
import statsmodels.api as sm
    # stats models mlr using ols
X_sm = X = sm.add_constant(X)

model = sm.OLS(y, X_sm)
model.fit().summary()
    # sklearn mlr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train, scoring= 'neg_mean_absolute_error'))

# lasso regression
from sklearn.linear_model import Lasso
lm_lasso = Lasso(tol=.1)
lm_lasso.fit(X_train, y_train)
np.mean(cross_val_score(lm_lasso, X_train, y_train, scoring= 'neg_mean_absolute_error')) # mean = -140619.3198

# lets find an optimal alpha for lasso
    # refer to plot_alpha.py
    # no alpha was found

# Random forest regression
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error')) # mean = -140619.3198

# tune models with Gridsearch CV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10, 100, 10), 'criterion':('mse', 'mae'), 'max_features':('auto', 'sqrt', 'log2')}

gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error')
gs.fit(X_train, y_train)

gs.best_score_ # best_score = -88073.2144
gs.best_estimator_  # n_estimators = 80

np.mean(cross_val_score(gs, X_train, y_train, scoring= 'neg_mean_absolute_error')) # mean =

# model with XGboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

xg_model = XGBRegressor()
xg_model.fit(X_train, y_train)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_test))) # mean = 81684.6751405997
np.mean(cross_val_score(xg_model, X_train, y_train, scoring= 'neg_mean_absolute_error')) # mean = -78391.39484302704
print(xg_model.score(X_train, y_train)) # score = 0.963939048440953
print(xg_model.score(X_test, y_test))   # score = 0.8332648142708925

# tune XGBoost model
xg_tune = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
xg_tune.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_test, y_test)],
             verbose=False)


np.mean(cross_val_score(xg_tune, X_train, y_train, scoring= 'neg_mean_absolute_error')) # mean = -74280.71830808994
print("Mean Absolute Error: " + str(mean_absolute_error(xgt_pred, y_test))) # mean = 84252.82728465475
print(xg_tune.score(X_train, y_train)) # score = 0.9376663289930381
print(xg_tune.score(X_test, y_test)) # score = 0.8267666279683534

xg_tune.best_iteration
xg_tune.best_score

# tune xgb model with gridsearchCV??
parameters_2 = {'n_estimators':range(10, 100, 10)}
parameters_3 = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [8],
              'min_child_weight': [11],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [1000], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}

gs_xg = GridSearchCV(xg_model, parameters_3, scoring='neg_mean_absolute_error',)
gs_xg.fit(X_train, y_train)

print(gs_xg.score(X_train, y_train))
print(gs_xg.score(X_test, y_test))


gs_xg.refit
gs_xg.score(X_train, y_train)
gs_xg.return_train_score

gs_xg.best_score_ # paramaters_2 = -78660.81728509977, parameters_3 = -74975.58453495518
gs_xg.best_estimator_ # parameters_2 = 90,


# test models

lm_pred = lm.predict(X_test)
lm_lasso_pred = lm_lasso.predict(X_test)
rf_pred = gs.best_estimator_.predict(X_test)
xg_model_pred = xg_model.predict(X_test)
xgt_pred = xg_tune.predict(X_test)
gs_xg_pred = gs_xg.best_estimator_.predict(X_test)


lm_pred_mae = mean_absolute_error(y_test, lm_pred)                  # mae = 145252.11391006556
lm_lasso_pred_mae = mean_absolute_error(y_test, lm_lasso_pred)      # mae = 145252.11391006556
rf_pred_mae = mean_absolute_error(y_test, rf_pred)                  # mae = 92353.81374913255
xg_model_pred_mae = mean_absolute_error(y_test, xg_model_pred)      # mae = 81684.6751405997
xgt_pred_mae = mean_absolute_error(y_test, xgt_pred)                # mae = 84252.82728465475
gs_xg_pred_mae = mean_absolute_error(y_test, gs_xg_pred)            # mae = 79812.05892175573  best

predictions = [lm_lasso_pred_mae, lm_lasso_pred_mae, rf_pred_mae, xg_model_pred_mae, xgt_pred_mae, gs_xg_pred_mae]

print(predictions)

print(min(predictions))



# productionization

import pickle
pickl = {'model': gs_xg.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ))

file_name = "FlaskAPI/model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

data_in = [5.0, 3.0, 2900.0, 6730.0, 1.0, 0.0, 8.0, 1830.0, 1070.0, 1977.0, 0.0, 98115.0, 2370.0, 6283.0]

model.predict(v)
#model.predict(s)

X_test.iloc[0:1].values

b = X_test.iloc[0:1]

p =
w = [[4.0000e+00, 2.2500e+00, 2.0700e+03, 8.8930e+03, 2.0000e+00,
        0.0000e+00, 8.0000e+00, 2.0700e+03, 0.0000e+00, 1.9860e+03,
        0.0000e+00, 9.8058e+04, 2.3900e+03, 7.7000e+03]]

v = pd.DataFrame(w)

v

v.columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
       'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15']
