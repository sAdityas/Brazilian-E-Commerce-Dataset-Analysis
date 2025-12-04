# %% [markdown]
# # Plan
# 1. Understand the Dataset Structure:
#     - This dataset contains multiple tables, just like a real company database.
#     - Important Tables
#         - `Orders.csv` : Each Order + status + stamps
#         - `order_items.csv`: Item wise detail of each order
#         - `products.cs`v: Product Details
#         - `Order.csv`: Order Detail
#         - `sellers.csv`: seller info
#         - `customers.csv`: customer info
#         - `order_payments.csv`: Payment Methods
#         - `orders_review.csv`: Customer review text + rating
#         - `geolocation.csv`: Lat/Long info
# 2. Identify Primary Relationships between table:
#     - `order_id` is the hub -> connects many tables
#     - `product_id` connects items -> products
#     - `seller_id` connects sellers -> seller
#     - `customer_id` connects order -> customers
#     - `order_id` connects orders -> payments -> reviews

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %% [markdown]
# # Libraries:
# - `pandas` : To analyze, clean, restructure and join different tables.
# - `matplotlib.pyplot`: To visualize data to make analysis and describing data easier.
# - `seaborn`: To visualize data and analyze data more deeply.

# %%
customers = pd.read_csv('./dataset/customers.csv')
geolocations = pd.read_csv('./dataset/geolocations.csv')
order_items = pd.read_csv('./dataset/order_items.csv')
order_payments = pd.read_csv('./dataset/order_payments.csv')
order_reviews = pd.read_csv('./dataset/customers.csv')
orders = pd.read_csv('./dataset/orders.csv')
pcnt = pd.read_csv('./dataset/product_category_name_translation.csv')
products = pd.read_csv('./dataset/products.csv')
sellers = pd.read_csv('./dataset/sellers.csv')


# %% [markdown]
# # Loading Data
# - Loading data into jupyter using pandas
#     - `pd.read_csv('file_name.csv')`
# - We have assigned the csv names are variables which makes it easier to understand, with which dataset are we working currently.

# %%
print(f'''
    Customer Table Columns: {customers.columns} \n
    geolocations Table Columns: {geolocations.columns} \n
    order_items Table Columns: {order_items.columns} \n
    order_payments Table Columns: {order_payments.columns} \n
    order_reviews Table Columns: {order_reviews.columns} \n
    orders Table Columns: {orders.columns} \n
    pcnt Table Columns: {pcnt.columns} \n
    products Table Columns: {products.columns} \n
    sellers Table Columns: {sellers.columns} \n
''')


# %% [markdown]
# # Understanding what columns exists
# - We have output all the column in each dataset table. 
# - This will help when are going to join tables and work with the table.

# %%
print(f'''
    {customers.info()} 
    {geolocations.info()} 
    {order_items.info()} 
    {order_payments.info()} 
    {order_reviews.info()} 
    {orders.info()} 
    {pcnt.info()} 
    {products.info()} 
    {sellers.info()} 
''')


# %% [markdown]
# # Table information
# - We have used the `info()` function to check the details of all tabel.
# - This shows:
#     - Length of data
#     - Dtype of the columns
#     - are they null or non-non
#     - Column names
#     - Number of Columns

# %% [markdown]
# - We found out that all the tables do not have any null value
# - As all the tables have columns with non-null type.

# %%
orders[['Date','Time']] = orders['order_purchase_timestamp'].str.split(' ',expand=True)

orders['Date'] = pd.to_datetime(orders['Date'])
orders['Time'] = pd.to_datetime(orders['Time'],format='%H:%M:%S').dt.time

orders['Date'].info()


# %% [markdown]
# # Type Conversion (orders table)
# 
# - Timestamp of orders was in object datatype.
# - This would have caused it to not work properly when setting up visual.
# - To change the dtype:
#     - `table['column'] = pd.to_datetime(table['column']).dt.date`
# - This will change the dtype from `object` to `datetime`.
# - But still this has both date and the time in single column. 
# - So we do:
#    - `table[['column1','column2']] = table['tosplit'].str.split(' ', expand=True)`
# - This is generate two new column with names given by spitting the timestamp column into two

# %%
df = orders.merge(order_payments, on='order_id', how='left')
order_total = df.groupby('order_id')['payment_value'].sum().reset_index()

order_outliers = orders.merge(order_total, on='order_id', how='left')

sns.boxplot(x=order_outliers['payment_value'])
plt.title('Order Payment Value Distribution')


# %% [markdown]
# # Outlier Check on Orders
# 
# 1. Most payments are very small
#     - The entire box is squeezed on the left side, indicating that:
#         - Majority if customers make low-value purchases.
#         - The typical order_value is far below 500rs
#     - This is Expected in marketplace with low-cost items.
# 2. Long tail of high-value orders:
#     - Multiple points appear far to the right, including values around:
#         - 3000
#         - 5000
#         - 7000
#         - 14000 
#     - These present:
#         - Bulk orders
#         - High ticket items
#         - Multi-item Purchase
#         - Customer Paying via multiple installments
# 3. Presence of multiple extreme outliers:
#     - The points far from the box indicate very high value payments.
#     - These are unusual comapared to the majority and may need futher investigation:
#         - Are these corporate buyers.
#         - Specific Product Categories.
#         - Sellers with expensive listing.

# %%
expensive_sellers = order_items.groupby('seller_id')[['seller_id','product_id','price']].head(100000).reset_index()
expensive_sellers_product = expensive_sellers.merge(products, on='product_id', how='inner')[['seller_id','price','product_category_name']].reset_index(drop=True)
expensive_sellers_product = expensive_sellers_product.merge(pcnt, on='product_category_name', how='left').drop('product_category_name', axis=1)

expensive_sellers.plot(kind='scatter', x='seller_id', y='price', title='Top Most Expensive Products Sold by Sellers')
plt.show()


# %% [markdown]
# # Expensive Sellers Scatter Plot
# 
# 1. Lower Range:
#     - Most of the items are under 1000rs range.
#     - These items maybe of daily use 
#         - items
#         - groceries
#         - stationary
#         - Low value accesories
# 2. Mid Range:
#     - The mid range if from 1000 and goes to like 3000.
#     - These are general use and semi luxury items like
#         - Home appliances.
#         - Fashion Items.
#         - Small Electronics.
#         - Mid-range gadgets
# 3. High Range:
#     - The hight range is from 3000 and almost to 7000.
#     - A noticeable number of points appear in this band.
#     - These items include:
#         - Premium Electronics
#         - Television
#         - Large Appliances.
#         - High quality furniture or decor
#  
# 

# %%
# 1. Make Month column (short name)
orders['Month'] = orders['Date'].dt.strftime('%b')

# 2. Aggregate: orders per month
month_counts = (
    orders.groupby('Month')['order_id']
          .count()
          .reset_index()
)

# 3. Set proper month order on the *aggregated* df
month_order = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

month_counts['Month'] = pd.Categorical(
    month_counts['Month'],
    categories=month_order,
    ordered=True
)

month_counts = month_counts.sort_values('Month')

# 4. Optional rename
month_counts = month_counts.rename(columns={'order_id': 'Total Orders'})

# 5. Plot from month_counts, not orders
plt.figure(figsize=(10,5))
sns.lineplot(
    data=month_counts,
    x='Month',
    y='Total Orders',
    marker='o',
    errorbar=None
)
plt.title("Orders per Month")
plt.xlabel("Month")
plt.ylabel("Total Orders")
plt.tight_layout()
plt.show()


# %% [markdown]
# # Orders Per Month
# - After analysis Orders on monthly basis we found out that most of the orders peak in month `Jun` and `Jul` a most. 
# - But there are periodic rise in orders every 2 months.
# - The reason behind `Jun` and `Jul` rise is the festivals that take place in brazil in `5th` `7th` or `8th` month.
# - During these months people tend to order festive items like `costumes`, `new accesories` , etc.
# - Once festive season is over orde rate tend to fall as no offers/deals are not available and no festivals.
# - We can try to launch new offers or deals in month of `Sept` to give orders a stable rise and fall and not these major falls.

# %%
order_items['revenue'] = order_items['freight_value'] + order_items['price']

orders_merged = orders.merge(order_items,on='order_id',how='inner')

total_revenue_Pm = orders_merged.groupby('Month')['revenue'].sum().reset_index()

total_revenue_Pm['Month']= pd.Categorical(
    total_revenue_Pm['Month'],
    categories=month_order,
    ordered=True
)

total_revenue_Pm = total_revenue_Pm.sort_values('Month')

plt.figure(figsize=(10,8))
plt.title('Total Revenue Each Month')

sns.lineplot(data=total_revenue_Pm, errorbar=None, x='Month', y='revenue',marker='s')
plt.xlabel('Month')
plt.ylabel('Revenue Generated')
plt.yticks()
plt.show()
print(total_revenue_Pm)



# %% [markdown]
# # Monthly Revenue Trend Analysis
# 
# 1. Strong Revenue Growth in Early Month:
#     - Jan -> May revenue rises steadily, indicating a period of increased customer spending, like due to seasonality or marketing campaigns.
# 2. Mid Year Stability:
#     - Revenue Remains Consistent high during June -> August.
#     - This suggest stable period with regular orders flow and strong sales performance.
# 3. Sharp Drop in September
#     - A significant drop can be seen in September, which may include:
#         - Seasional Dip
#         - Fewer Promotions
#         - Operational / Logistics Issue
#         - Order Cancellatios or Delays.
# 4. Late Year Recoverly:
#     - Revenue Starts improving again as start of october and reaches a small peak in november, likely due to:
#         - Festive season
#         - Holiday Shopping Trends
#         - Discount Campaigns
# 5. Slight drop in december:
#     - There is a small decline in month of december which could relate to end of year stock issues or shipping slowdowns.
# 

# %%
best_selling_categories = order_items.merge(products, on='product_id', how='inner')
best_selling_categories = best_selling_categories.groupby('product_category_name')['order_item_id'].count().reset_index()
best_selling_categories = best_selling_categories.sort_values(by='order_item_id', ascending=False).head(10)

best_selling_categories = best_selling_categories.merge(pcnt, left_on='product_category_name', right_on='product_category_name', how='left')
best_selling_categories = best_selling_categories[['product_category_name_english', 'order_item_id']]
best_selling_categories.columns = ['product_category_name', 'Total Orders']
plt.figure(figsize=(12,6))
sns.barplot(data=best_selling_categories, x='product_category_name', y='Total Orders', palette='viridis', hue='Total Orders', legend=False)
plt.xticks(rotation=60)
plt.title('Top 10 Best Selling Product Categories')
plt.show()
print(best_selling_categories.head())


# %% [markdown]
# # Top 10 Best Selling Categories of Products
# 
# 1. This visualization shows top 10 product categories based on the number of orders being placed.
#     - `bed-bath-table` has the most sales, as this contains products with cheap rate and products that are needed to buy periodically.
#     - such as:
#         - bed sheet
#         - cushions
#         - blankets
#         - towels
#         - bath mats
#         - table cloths
#         - napkins, etc.
#     - `auto` has least amount of reorders and sales. These are the items which we buy once in few years or months, these items has no need to be bought every 2 3 months or so.
#     - such as:
#         - Seat covers
#         - Car cleaning kits
#         - helmet visors
#         - wrenches, tool kits
#         - car charger devices, etc
# 

# %%
best_selling_categories = order_items.merge(products, on='product_id', how='inner')

cat_sales = (
    best_selling_categories
    .groupby('product_category_name')['order_item_id']
    .count()
    .reset_index()
    .rename(columns={'order_item_id': 'total_sales'})
    .merge(pcnt, on='product_category_name', how='left')
)

top_cat = (
    cat_sales
    .sort_values(by='total_sales', ascending=False)
    .head(10)
)
top_cat_keys = top_cat['product_category_name'].tolist()

bottom_cat = (
    cat_sales
    .sort_values(by='total_sales', ascending=True)
    .head(10)
)
bottom_cat_keys = bottom_cat['product_category_name'].tolist()

pricecomp = best_selling_categories[
    best_selling_categories['product_category_name'].isin(top_cat_keys + bottom_cat_keys)
].copy()

import numpy as np

pricecomp['Category Type'] = np.where(
    pricecomp['product_category_name'].isin(top_cat_keys),
    'Top 10',
    'Bottom 10'
)

# optional: bring English category name
pricecomp = pricecomp.merge(pcnt, on='product_category_name', how='left')


# ==== AVG PRICE PER CATEGORY (IF YOU WANT IT) ====
avg_pricecomp = (
    pricecomp
    .groupby(['product_category_name', 'Category Type'])['price']
    .mean()
    .reset_index()
    .merge(pcnt, on='product_category_name', how='left')
)


# ==== BOX PLOT: BEST vs LEAST SELLING PRICE DISTRIBUTION ====
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=pricecomp,
    x='Category Type',   # <-- FIXED HERE
    y='price'
)
plt.title('Price Distribution: Best vs Least Selling Categories')
plt.ylabel('Price')
plt.xlabel('Category Group')
plt.show()



