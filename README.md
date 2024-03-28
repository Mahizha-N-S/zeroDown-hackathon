# zeroDown-hackathon
# Problem Statement 1
PriceProbe: Predicting Property Values
In the dynamic realm of US real estate, accurate pricing stands as the cornerstone of successful
transactions. With the market constantly evolving and property values fluctuating, the ability to
determine fair and competitive prices is paramount. Using the partial raw market data
provided, your task is to predict home prices for properties listed for sale, by progressing
through the following milestones.
Milestones:
1. ERD: Add entity relationship diagram based on DDL statements provided.
2. EDA:
● Range of attributes
● Geographical spread
● Temporal spread
● Identify outlier homes and homes with incorrect data
3. Homes Deduplication: Devise an scalable algorithm to identify duplicate homes.
Duplicates can be classified into 2 types, identify both separately.
● Absolute duplicate - same home, listed in the market at almost same time.
● Pseudo duplicate - same home, listed at different points in time.
4. Home Comparables: Given a home id, devise an algorithm to provide a list of similar
homes.
5. Price Estimation: Given home attributes(bed, bath, city/zipcode etc...) estimate price
based on sold homes.
Note:
1. For comparables & price estimation, bed, bath, city/zipcode will be mandatory inputs.
Other attributes like finished_sqft, lot_size_sqft, home_type, etc… can be optional
inputs.
2. Preferably, use PostgreSQL database.
Data:
home_info.sql
market.sql
market_geom.sql

# My Solution
+ Each milestone is put into separate folder along with the output docx
+ And a flak application is present in the folder display_all_output
+ run in your loaclhost the program sol.py
+ in sol.py we have buttons for each milestone , you can choose your needed milestone to view your output results
+ to see the milestone 2, run the sol.py in your local host with http://127.0.0.1:5000/milestone2
+ and for next milestone 3, http://127.0.0.1:5000/milestone3



