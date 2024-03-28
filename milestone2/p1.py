#mile stone 2
#Exploratory Data Analysis (EDA) on the home_info table
from flask import Flask, render_template
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


conn = psycopg2.connect(host="localhost",dbname="zeroDown",user="postgres",password="mahizha",port=5432)
cur = conn.cursor()

#cur.execute("""SELECT * FROM home_info;""")
#print(cur.fetchone())

# Load data by the sql query and save it in df
query = "SELECT * FROM home_info;"
df= pd.read_sql_query(query, conn)

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



plt.figure(figsize=(10, 8))
sns.boxplot(x='bedrooms', y='finished_sqft', data=df)
plt.title("Bedrooms vs. Finished Sqft")
plt.show()

# Temporal Spread
plt.figure(figsize=(12, 6))
sns.histplot(df['listing_contract_date'], bins=30, kde=True)
plt.title('Temporal Spread of Listings')
plt.xlabel('Listing Contract Date')
plt.ylabel('Count')
plt.show()

# homes with incorrect data
# For example, check for missing values
print("Number of missing values per column:")
print(df.isnull().sum())



#------to identify the outlier homes using the Isolation Forest algorithm------

# here i choose these features for outlier detection
features = ['finished_sqft', 'bedrooms', 'bathrooms', 'listing_price']
X = df[features]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Fit Isolation Forest model to detect outliers
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(X_imputed)

# Predict outliers
outliers = clf.predict(X_imputed)

# Add outlier predictions to the DataFrame class->(1,-1)
df['outlier'] = outliers

# Display homes identified as outliers -> -1
outlier_homes = df[df['outlier'] == -1]
print("Outlier Homes:")
print(outlier_homes[['id', 'finished_sqft', 'listing_price']])

# Plot the outliers
plt.figure(figsize=(10, 8))
sns.scatterplot(x='finished_sqft', y='listing_price', hue='outlier', data=df)
plt.title("Outliers Detected by Isolation Forest")
plt.xlabel("Finished Sqft")
plt.ylabel("Listing Price")

# Annotate outlier homes with their IDs
for index, row in outlier_homes.iterrows():
    plt.annotate(row['id'], (row['finished_sqft'], row['listing_price']), textcoords="offset points", xytext=(0,10), ha='center')

plt.legend(title='Outlier', loc='upper right')
plt.show()







# Identify Outliers and Incorrect Data
# Example here: Checking for outliers in 'finished_sqft' column
Q1 = df['finished_sqft'].quantile(0.25)
Q3 = df['finished_sqft'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['finished_sqft'] < (Q1 - 1.5 * IQR)) | (df['finished_sqft'] > (Q3 + 1.5 * IQR))]
print("Outliers in 'finished_sqft' column:")
print(outliers)


cur.close()
conn.close()