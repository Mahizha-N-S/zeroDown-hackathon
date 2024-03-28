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





def estimate_price(home_id, bed, bath, city, zipcode, **kwargs):
    # Load data of sold homes
    # `df_sold_homes` contains the data of sold homes with columns ['bedrooms', 'bathrooms', 'city_market_id', 'zipcode_market_id', 'listing_price']

    # Filter sold homes based on mandatory inputs
    filtered_homes = df_home_info[(df_home_info['bedrooms'] == bed) & 
                                   (df_home_info['bathrooms'] == bath) & 
                                   (df_home_info['city_market_id'] == city) & 
                                   (df_home_info['zipcode_market_id'] == zipcode)]

    # Filter further based on optional inputs
    for key, value in kwargs.items():
        if key in df_home_info.columns:
            filtered_homes = filtered_homes[abs(filtered_homes[key] - value) <= 5]  # Assuming a tolerance of 5 for similarity

    # Calculate similarity score
    filtered_homes['similarity_score'] = 1  # Placeholder for now, actual calculation needed

    # Calculate weighted average of listing prices
    total_similarity = filtered_homes['similarity_score'].sum()
    if total_similarity == 0:
        return "No comparable homes found."
    weighted_prices = filtered_homes['listing_price'] * filtered_homes['similarity_score']
    estimated_price = weighted_prices.sum() / total_similarity

    return estimated_price

# Example Usage
home_id = 186
bed = 2
bath = 2
city = 714
zipcode = 1687
optional_attributes = {'finished_sqft': 1161, 'lot_size_sqft': 48941}
estimated_price = estimate_price(home_id, bed, bath, city, zipcode, **optional_attributes)
print("Estimated Price:", estimated_price)


#using linear regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


# Selecting relevant columns for the model
features = ['bedrooms', 'bathrooms', 'city_market_id', 'zipcode_market_id']
target = 'listing_price'

# Combining both datasets to include sold homes in the training data
df_sold_homes = df_home_info[df_home_info['status'] == 'SOLD']

df_train = pd.concat([df_home_info, df_sold_homes])

# Splitting the data into training and test sets
X = df_train[features]
y = df_train[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Training the linear regression model
model = LinearRegression()
model.fit(X_train_imputed, y_train)

# Making predictions
y_pred = model.predict(X_test_imputed)

# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Example usage: Predicting the price of a new home
new_home = pd.DataFrame([[2, 2, 714, 1687]], columns=['bedrooms', 'bathrooms', 'city_market_id', 'zipcode_market_id'])
new_home_imputed = imputer.transform(new_home)
predicted_price = model.predict(new_home_imputed)
print(f"Predicted Price for New Home: {predicted_price}")


conn.close()
cur.close()