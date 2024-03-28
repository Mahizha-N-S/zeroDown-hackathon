#milestone 5
#Price Estimation: Given home attributes(bed, bath, city/zipcode etc...) estimate price based on sold homes.
#using linear regression

import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt

conn = psycopg2.connect(host="localhost",dbname="zeroDown",user="postgres",password="mahizha",port=5432)
cur = conn.cursor()

query = "SELECT * FROM home_info;"
cur.execute(query)
rows = cur.fetchall()    
df_home_info= pd.DataFrame(rows, columns=['id', 'listing_key', 'source_system', 'address', 'usps_address', 'status', 'listing_contract_date', 'on_market_date', 'pending_date', 'last_sold_date', 'off_market_date', 'original_listing_price', 'listing_price', 'last_sold_price', 'home_type', 'finished_sqft', 'lot_size_sqft', 'bedrooms', 'bathrooms', 'year_built', 'new_construction', 'has_pool', 'state_market_id', 'city_market_id', 'zipcode_market_id', 'neighborhood_level_1_market_id', 'neighborhood_level_2_market_id', 'neighborhood_level_3_market_id', 'long', 'lat', 'crawler'])

query = "SELECT * FROM market;"
cur.execute(query)
rows = cur.fetchall()
df_market=pd.DataFrame(rows,columns=['id','name','market_level','state','city','zipcode','neighborhood','neighborhood_source'])

query = "SELECT * FROM market_geom;"
cur.execute(query)
rows = cur.fetchall()
df_market_geom=pd.DataFrame(rows,columns=['id','market_id','longitude','latitude','geom','area_in_sq_mi','centroid_geom'])





# Merge home and market data
df_merged = pd.merge(df_home_info, df_market, left_on='city_market_id', right_on='id', how='inner')
df_merged = pd.merge(df_merged, df_market_geom, on='market_id', how='inner')

# Prepare feature columns (mandatory inputs)
features = ['bedrooms', 'bathrooms', 'city_market_id', 'zipcode_market_id']

# Add optional input features if available
optional_features = ['finished_sqft', 'lot_size_sqft', 'home_type']
features.extend(optional_features)

# Remove rows with missing values in any of the feature columns
df_merged = df_merged.dropna(subset=features)

# Use only the selected features
X = df_merged[features]

# Target variable for price estimation
y = df_merged['listing_price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict prices on the test data
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Example usage to estimate price for a new home
new_home_features = {
    'bedrooms': 3,
    'bathrooms': 2,
    'city_market_id': 714,
    'zipcode_market_id': 1687,
    'finished_sqft': 1500,
    'lot_size_sqft': 6000,
    'home_type': 'SingleFamilyResidence'
}
new_home_features_df = pd.DataFrame([new_home_features])
predicted_price = model.predict(new_home_features_df)[0]
print("Predicted Price for the New Home:", predicted_price)





conn.close()
cur.close()