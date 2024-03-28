#mile stone 4

#Home Comparables: Given a home id, devise an algorithm to provide a list of similar homes
import pandas as pd
import psycopg2

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean

conn = psycopg2.connect(host="localhost",dbname="zeroDown",user="postgres",password="mahizha",port=5432)
cur = conn.cursor()

query = "SELECT * FROM home_info;"
cur.execute(query)
rows = cur.fetchall()    
df= pd.DataFrame(rows, columns=['id', 'listing_key', 'source_system', 'address', 'usps_address', 'status', 'listing_contract_date', 'on_market_date', 'pending_date', 'last_sold_date', 'off_market_date', 'original_listing_price', 'listing_price', 'last_sold_price', 'home_type', 'finished_sqft', 'lot_size_sqft', 'bedrooms', 'bathrooms', 'year_built', 'new_construction', 'has_pool', 'state_market_id', 'city_market_id', 'zipcode_market_id', 'neighborhood_level_1_market_id', 'neighborhood_level_2_market_id', 'neighborhood_level_3_market_id', 'long', 'lat', 'crawler'])

query = "SELECT * FROM market;"
cur.execute(query)
rows = cur.fetchall()
df_market=pd.DataFrame(rows,columns=['id','name','market_level','state','city','zipcode','neighborhood','neighborhood_source'])

query = "SELECT * FROM market_geom;"
cur.execute(query)
rows = cur.fetchall()
df_market_geom=pd.DataFrame(rows,columns=['id','market_id','longitude','latitude','geom','area_in_sq_mi','centroid_geom'])

total=[]


#to make the simlarity score in the range of 0-100 %
def scale_similarity_score(total_similarity):
    min_val = total_similarity.min()
    max_val = total_similarity.max()
    scaled_similarity = (total_similarity - min_val) / (max_val - min_val)
    scaled_similarity *= 100
    return scaled_similarity


#this function takes the mandatory inputs bed, bath, city/zipcode and also the optional inputs like finished_sqft, lot_size_sqft, home_type, etc…
def find_home_comparables(home_id, bed, bath, city, zipcode, **kwargs):
    #mandatory inputs
    if not all([bed, bath, city, zipcode]):
        return "Mandatory inputs (bed, bath, city, zipcode) are required."

    #here we filter homes based on mandatory attributes
    # the similarity score is calculated based on whether the values of these attributes match exactly between the input home and other homes
    #If a home meets all mandatory criteria, it gets a perfect score for these attributes
    #the similarity score is implicitly binary: either 1 (for a match) or 0 (for no match)
    filtered_df = df[(df['bedrooms'] == bed) & (df['bathrooms'] == bath) & (df['city_market_id'] == city) & (df['zipcode_market_id'] == zipcode)]

    #here we iterate thorugh all the attributes and if found the absolute difference is calculated
    for key, value in kwargs.items():
        if key in df.columns:
            filtered_df[key + '_similarity'] = abs(filtered_df[key] - value)

    #similarity score
    filtered_df['total_similarity'] = filtered_df.filter(like='_similarity').sum(axis=1)
    
    # making the score in 0-100 range
    filtered_df['total_similarity_scaled'] = scale_similarity_score(filtered_df['total_similarity'])
    
    # Sort homes by total similarity
    comparable_homes = filtered_df.sort_values('total_similarity_scaled')[['id', 'bedrooms', 'bathrooms', 'city_market_id', 'zipcode_market_id', 'total_similarity_scaled']]
    totals=filtered_df['total_similarity_scaled'].tolist()
    return comparable_homes,totals


# Example Usage
home_id = 186
bed = 2
bath = 2
city = 714
zipcode = 1687
optional_attributes = {'finished_sqft': 1161, 'lot_size_sqft': 48941}

comparables,total = find_home_comparables(home_id, bed, bath, city, zipcode, **optional_attributes)
print(comparables)
conn.close()


#histogram chart
plt.figure(figsize=(12, 6))
plt.hist(total, bins=10, range=(0, 100), color='skyblue', edgecolor='black')
plt.xlabel('Total Similarity (%)')
plt.ylabel('Frequency')
plt.title('Histogram of Total Similarity Values')
plt.grid(axis='y', alpha=0.75)
plt.xticks(range(0, 101, 10)) #x axis in the percentage score
plt.show()

