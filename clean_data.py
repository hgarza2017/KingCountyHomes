
import pandas as pd

df = pd.read_csv('kc_house_data.csv')

# remove trialing zeros from price
price = df['price'].apply(lambda x: int(round(x)))
df['price'] = price

# remove trailing zeros from bathrooms
df['bathrooms'].dtype
dec = 2
bath = df['bathrooms'].apply(lambda x: round(x, dec))
df['bathrooms'] = bath
df['bathrooms'].head()

# remove trailing zeros from floors
floors = df['floors'].apply(lambda x: round(x, dec))
df['floors'] = floors

# save cleaned data
df.to_csv('kc_house_data_cleaned.csv')
