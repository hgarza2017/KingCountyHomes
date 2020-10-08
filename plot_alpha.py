import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score


df = pd.read_csv('kc_house_data_cleaned.csv')

# choose columns to use
df_model = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'grade', 'sqft_above', 'sqft_basement',
               'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15']]


# train test split

X = df_model.drop('price', axis = 1)
y = df_model.price.values
#ya = df_model.price this is the series which we shouldn't use

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# multiple linear regresssion
    # stats models mlr using ols
X_sm = X = sm.add_constant(X)

model = sm.OLS(y, X_sm)
model.fit().summary()

    # sklearn mlr

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train, scoring= 'neg_mean_absolute_error'))

# lasso regression
lm_lasso = Lasso(tol=.1)

np.mean(cross_val_score(lm_lasso, X_train, y_train, scoring= 'neg_mean_absolute_error'))

# lets find an optimal alpha for lasso
alpha = []
error = []

for i in range(1, 10000):
    alpha.append(i/10000)
    lmLasso = Lasso(alpha=(i/10000), tol=.1)
    error.append(np.mean(cross_val_score(lmLasso, X_train, y_train, scoring= 'neg_mean_absolute_error')))

plt.plot(alpha, error)
plt.show()
