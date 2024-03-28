#Absolute Duplicate Detection: homes are same as they are listed at the same time
#Pseudo Duplicate Detection: dentify homes with similar attributes but listed at different points in time.
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from sqlalchemy import create_engine

conn = psycopg2.connect(host="localhost",dbname="zeroDown",user="postgres",password="mahizha",port=5432)
cur = conn.cursor()



#df is the data frame containing all the home_info data

query = "SELECT * FROM home_info;"
cur.execute(query)


rows = cur.fetchall()


cur.close()
conn.close()

    
df = pd.DataFrame(rows, columns=['id', 'listing_key', 'source_system', 'address', 'usps_address', 'status', 'listing_contract_date', 'on_market_date', 'pending_date', 'last_sold_date', 'off_market_date', 'original_listing_price', 'listing_price', 'last_sold_price', 'home_type', 'finished_sqft', 'lot_size_sqft', 'bedrooms', 'bathrooms', 'year_built', 'new_construction', 'has_pool', 'state_market_id', 'city_market_id', 'zipcode_market_id', 'neighborhood_level_1_market_id', 'neighborhood_level_2_market_id', 'neighborhood_level_3_market_id', 'long', 'lat', 'crawler'])




def detect_absolute_duplicates(df):
    # Group by unique identifiers or attributes
    grouped = df.groupby(['address', 'bedrooms', 'bathrooms', 'home_type'])
    
    # Calculate time interval threshold as we check on the same day we set this threshold
    time_threshold = timedelta(days=1)
    
    # this  list stores absolute duplicate pairs
    absolute_duplicates = []

    for _, group in grouped:
       
        group = group.sort_values('listing_contract_date')
        # Calculate time difference between consecutive listings to check the time in market
        group['time_diff'] = group['listing_contract_date'].diff()
        # Flag rows where time difference is less than threshold
        duplicates = group[group['time_diff'] <= time_threshold]
        if len(duplicates) > 1:
            # here we insert the duplicate pairs
            absolute_duplicates.extend([(row1['id'], row2['id']) for index1, row1 in duplicates.iterrows() 
                                         for index2, row2 in duplicates.iterrows() if index1 < index2])

    return absolute_duplicates

def detect_pseudo_duplicates(df):
    # Group by unique identifiers or attributes
    grouped = df.groupby(['address', 'bedrooms', 'bathrooms', 'home_type'])
    #here no such threshold like the previous
    #this list stores pseudo duplicate pairs
    pseudo_duplicates = []

    for _, group in grouped:
       
        group = group.sort_values('listing_contract_date')
        # Flag rows where listing_contract_date is duplicated
        duplicates = group[group.duplicated('listing_contract_date', keep=False)]
        if len(duplicates) > 1:
           
            pseudo_duplicates.extend([(row1['id'], row2['id']) for index1, row1 in duplicates.iterrows() 
                                       for index2, row2 in duplicates.iterrows() if index1 < index2])

    return pseudo_duplicates



# Identify Absolute Duplicates
#the returned are the home_info id pairs which we find duplicate
absolute_duplicates = detect_absolute_duplicates(df)
print("Absolute Duplicates:")
print(absolute_duplicates)

# Identify Pseudo Duplicates
pseudo_duplicates = detect_pseudo_duplicates(df)
print("Pseudo Duplicates:")
print(pseudo_duplicates)
