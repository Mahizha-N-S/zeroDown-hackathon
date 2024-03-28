#mile stone 2
#Exploratory Data Analysis (EDA) on the home_info table

import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

conn = psycopg2.connect(host="localhost",dbname="zeroDown",user="postgres",password="mahizha",port=5432)
cur = conn.cursor()

#cur.execute("""SELECT * FROM home_info;""")
#print(cur.fetchone())

# Load data by the sql query and save it in df
query = "SELECT * FROM home_info;"
df = pd.read_sql_query(query, conn)

# Range of attributes
numeric_cols = ['finished_sqft', 'bedrooms', 'bathrooms', 'listing_price']
categorical_cols = ['home_type', 'status']

print("Descriptive Statistics for Numeric Attributes:")
print(df[numeric_cols].describe())

#here i display the unique values in the categorical column
print("\nUnique Values for Categorical Attributes:")
for col in categorical_cols:
    print(col, df[col].unique())


# Geographical spread
#checking if lat and long is present and plot to visualize the geo spread
if 'long' in df.columns and 'lat' in df.columns:
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='long', y='lat', data=df)
    plt.title("Geographical Spread of Homes")
    plt.show()
else:
    print("Longitude and latitude columns not found in the DataFrame.")

# Temporal spread
temporal_cols = ['listing_contract_date', 'on_market_date', 'pending_date', 'last_sold_date', 'off_market_date']
for col in temporal_cols:
    df[col] = pd.to_datetime(df[col])
    print(f"Min {col}: {df[col].min()}, Max {col}: {df[col].max()}")

#here we identify outlier homes
plt.figure(figsize=(10, 8))
sns.boxplot(x='bedrooms', y='finished_sqft', data=df)
plt.title("Bedrooms vs. Finished Sqft")
plt.show()

# homes with incorrect data
# For example, check for missing values
print("Number of missing values per column:")
print(df.isnull().sum())