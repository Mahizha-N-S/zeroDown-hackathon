#mile stone 2
#Exploratory Data Analysis (EDA) on the home_info table
from flask import Flask, render_template
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import base64
from io import BytesIO

app = Flask(__name__, template_folder='templates')


@app.route('/milestone2')
def home1():
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
        geo_spread_img = BytesIO()
        plt.savefig('static/geo_spread.png')
        geo_spread_img_b64 = base64.b64encode(geo_spread_img.getvalue()).decode('utf-8')
        plt.close()
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
    plt.savefig('static/bed.png')  # Save the image
    plt.close()
    
    # Temporal Spread
    plt.figure(figsize=(12, 6))
    sns.histplot(df['listing_contract_date'], bins=30, kde=True)
    plt.title('Temporal Spread of Listings')
    plt.xlabel('Listing Contract Date')
    plt.ylabel('Count')
    plt.savefig('static/temporal_spread.png')  # Save the image
    plt.close()
    
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
    
    return render_template('eda_result.html', df=df, numeric_cols=numeric_cols, categorical_cols=categorical_cols, outlier_homes=outlier_homes)

#-----------------------------------------------------------

@app.route('/milestone3')
def home2():
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

    return render_template('duplicate.html', absolute_duplicates=absolute_duplicates, pseudo_duplicates=pseudo_duplicates)
    


if __name__ == '__main__':
    app.run(debug=True)