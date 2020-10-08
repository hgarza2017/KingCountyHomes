
import numpy as np
import pandas as pd

data = [[4.0000e+00, 2.2500e+00, 2.0700e+03, 8.8930e+03, 2.0000e+00,
        0.0000e+00, 8.0000e+00, 2.0700e+03, 0.0000e+00, 1.9860e+03,
        0.0000e+00, 9.8058e+04, 2.3900e+03, 7.7000e+03]]

data_df = pd.DataFrame(data)

data_df.columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15']

data_in = data_df
