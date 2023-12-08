#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().system('pip install pymssql')


# In[3]:


# MySQL Database connection
import pymysql
import pandas as pd
from sqlalchemy import create_engine

data = pd.read_csv(r"C:\Users\nikhil\Downloads\DataCoSupplyChainDataset.csv",encoding='cp1252')
data.to_csv(r"C:\\Users\nikhil\Downloads\datax.csv")
# Creating engine which connect to MySQL
user = 'root' # user name
pw = 'Nikhil@123' # password
db = 'sales' # database

import urllib.parse
pw = urllib.parse.quote_plus("Nikhil@123")  # '123%40456'

# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# dumping data into database 
data.to_sql('sales', con = engine, if_exists = 'replace', chunksize = 500, index = False)

# loading data from database
#sql = 'select * from sales'

#edu = pd.read_sql_query(sql, con = engine)

#print(edu)


# In[4]:


# loading data from database
sql = 'select * from sales'
df = pd.read_sql_query(sql, con = engine)

print(df)


# In[5]:


df.head()


# In[6]:


pd.set_option('display.max_columns', None)
df.head()


# In[7]:


"""FIELDS,DESCRIPTION
Type,:  Type of transaction made
Days for shipping (real)     ,:  Actual shipping days of the purchased product
Days for shipment (scheduled),:  Days of scheduled delivery of the purchased product
Benefit per order,:  Earnings per order placed
Sales per customer,:  Total sales per customer made per customer
Delivery Status,":  Delivery status of orders: Advance shipping , Late delivery , Shipping canceled , Shipping on time"
Late_delivery_risk           ,":  Categorical variable that indicates if sending is late (1), it is not late (0)."
Category Id,:  Product category code
Category Name,:  Description of the product category
Customer City,:  City where the customer made the purchase
Customer Country,:  Country where the customer made the purchase
Customer Email,:  Customer's email
Customer Fname,:  Customer name
Customer Id,:  Customer ID
Customer Lname,:  Customer lastname
Customer Password,:  Masked customer key
Customer Segment,":  Types of Customers: Consumer , Corporate , Home Office"
Customer State,:  State to which the store where the purchase is registered belongs
Customer Street,:  Street to which the store where the purchase is registered belongs
Customer Zipcode,:  Customer Zipcode
Department Id,:  Department code of store
Department Name,:  Department name of store
Latitude,:  Latitude corresponding to location of store
Longitude,:  Longitude corresponding to location of store
Market,":  Market to where the order is delivered : Africa , Europe , LATAM , Pacific Asia , USCA"
Order City,:  Destination city of the order
Order Country,:  Destination country of the order
Order Customer Id,:  Customer order code
order date (DateOrders),:  Date on which the order is made
Order Id,:  Order code
Order Item Cardprod Id,:  Product code generated through the RFID reader
Order Item Discount,:  Order item discount value
Order Item Discount Rate     ,:  Order item discount percentage
Order Item Id,:  Order item code
Order Item Product Price     ,:  Price of products without discount
Order Item Profit Ratio,:  Order Item Profit Ratio
Order Item Quantity,:  Number of products per order
Sales,:  Value in sales
Order Item Total  ,:  Total amount per order
Order Profit Per Order,:  Order Profit Per Order
Order Region,":  Region of the world where the order is delivered :  Southeast Asia ,South Asia ,Oceania ,Eastern Asia, West Asia , West of USA , US Center , West Africa, Central Africa ,North Africa ,Western Europe ,Northern , Caribbean , South America ,East Africa ,Southern Europe , East of USA ,Canada ,Southern Africa , Central Asia ,  Europe , Central America, Eastern Europe , South of  USA "
Order State,:  State of the region where the order is delivered
Order Status,":  Order Status : COMPLETE , PENDING , CLOSED , PENDING_PAYMENT ,CANCELED , PROCESSING ,SUSPECTED_FRAUD ,ON_HOLD ,PAYMENT_REVIEW"
Product Card Id,:  Product code
Product Category Id,:  Product category code
Product Description,:  Product Description
Product Image,:  Link of visit and purchase of the product
Product Name,:  Product Name
Product Price,:  Product Price
Product Status,":  Status of the product stock :If it is 1 not available , 0 the product is available "
Shipping date (DateOrders)   ,:  Exact date and time of shipment
Shipping Mode,":  The following shipping modes are presented : Standard Class , First Class , Second Class , Same Day"""


# In[7]:


data=df.copy()
FeatureList=['Type', 'Benefit per order', 'Sales per customer', 
          'Delivery Status', 'Late_delivery_risk', 'Category Name', 'Customer City', 'Customer Country', 
             'Customer Id', 'Customer Segment', 
          'Customer State', 'Customer Zipcode', 'Department Name', 'Latitude', 'Longitude',
          'Market', 'Order City', 'Order Country', 'Order Customer Id', 'order date (DateOrders)', 'Order Id', 
          'Order Item Cardprod Id', 'Order Item Discount', 'Order Item Discount Rate', 'Order Item Id', 
          'Order Item Product Price', 'Order Item Profit Ratio', 'Order Item Quantity', 'Sales', 'Order Item Total', 
          'Order Profit Per Order', 'Order Region', 'Order State', 'Order Status', 'Product Card Id',
          'Product Category Id', 'Product Image', 'Product Name', 'Product Price', 'Product Status',
       'shipping date (DateOrders)', 'Shipping Mode']

df1=df[FeatureList]
df1.head()


# In[8]:


"""Exploratory Data Analysis"""


# In[9]:


df1.dtypes


# In[10]:


df1.isna().sum()


# In[11]:


duplicate = df1['Order Id'].duplicated()  # Returns Boolean Series denoting duplicate rows.
duplicate

sum(duplicate)


# In[12]:


dff=df1.iloc[:100000,20]
dff


# In[13]:


duplicate = dff.duplicated()  # Returns Boolean Series denoting duplicate rows.
duplicate

sum(duplicate)


# In[ ]:





# In[14]:


num_features=[column for column in df1.columns if df1[column].dtype != "O"]


# In[15]:


num_features


# In[16]:


#outlier detection part

fig=plt.figure(figsize=(20,20))

for i in range(len(num_features)):
    plt.subplot(5,5,i+1)
    sns.boxplot(data=data,x=data[num_features[i]])

plt.tight_layout()
plt.show()
     


# In[17]:


df1.describe()


# # Sales analysis

# In[18]:


#order country
df_sales_country=df1.groupby([ 'Order Country'])['Sales'].sum().reset_index(name='Sales of Orders').sort_values(by= 'Sales of Orders', ascending= False)
sns.barplot(data=df_sales_country.head(10), x='Sales of Orders',y = 'Order Country')


# In[29]:


#top 10 customers sales
top_sales_cust=df1.groupby(['Customer Id',])['Sales'].sum().reset_index(name='sales').sort_values(by='sales',ascending = False)
sns.barplot(data=top_sales_cust.head(10),x='Customer Id',y='sales')


# In[20]:


#product
df_sales_country=df1.groupby([ 'Product Name'])['Sales'].sum().reset_index(name='Sales of Orders').sort_values(by= 'Sales of Orders', ascending= False)
sns.barplot(data=df_sales_country.head(10), x='Sales of Orders',y = 'Product Name')


# In[21]:


#Product and deliveray status
df_sales_pd=df1.groupby([ 'Product Name', 'Delivery Status'])['Sales'].sum().reset_index(name='Sales of Orders').sort_values(by= 'Sales of Orders', ascending= False)
sns.barplot(data=df_sales_pd.head(10), x='Sales of Orders',y = 'Product Name')


# In[22]:


#Product and order region
df_sales_pr=df1.groupby([ 'Product Name', 'Order Region'])['Sales'].sum().reset_index(name='Sales of Orders').sort_values(by= 'Sales of Orders', ascending= False)
sns.barplot(data=df_sales_pr.head(100), x='Sales of Orders',y = 'Product Name')
#sns.countplot(data=df_sales_pr.head(10), x='Sales of Orders')


# In[23]:


#'Type of payment
df_sales_pr=df1.groupby([ 'Type'])['Sales'].sum().reset_index(name='Sales of Orders').sort_values(by= 'Sales of Orders', ascending= False)
sns.barplot(data=df_sales_pr.head(10), x='Sales of Orders',y = 'Type')
#sns.countplot(data=df_sales_pr.head(10), x='Sales of Orders')


# # top 10 customers by most order placing

# In[33]:


df1['Customer_ID_STR']=df1['Customer Id'].astype(str)
data_customers=df1.groupby(['Customer_ID_STR'])['Order Id'].count().reset_index(name='Number of Orders').sort_values(by= 'Number of Orders', ascending= False)
sns.barplot(data=data_customers.head(10),x='Number of Orders', y='Customer_ID_STR'  )


# In[39]:


df1['Customer_id']=df1['Customer Id'].astype(str)
data_customers=df1.groupby(['Customer_id'])['Sales'].sum().reset_index(name='sales').sort_values(by= 'sales', ascending= False)
sns.barplot(data=data_customers.head(10),x='sales', y='Customer_id'  )


# In[38]:


df1['Customer_Id']


# In[ ]:


df1


# #  top 10 customers by most profit

# In[25]:


df1['Customer_ID_STR']=df1['Customer Id'].astype(str)
data_customers_profit=df1.groupby(['Customer_ID_STR'])['Order Profit Per Order'].sum().reset_index(name='Profit of Orders').sort_values(by= 'Profit of Orders', ascending= False)
sns.barplot(data=data_customers_profit.head(10),x='Profit of Orders', y='Customer_ID_STR')


# In[26]:


data_Customer_Segment=df1.groupby(['Customer Segment'])['Order Id'].count().reset_index(name='Number of Orders').sort_values(by= 'Number of Orders', ascending= False)


# In[27]:


#Category Name
data_Category_Name=df1.groupby(['Category Name'])['Order Id'].count().reset_index(name='Number of Orders').sort_values(by= 'Number of Orders', ascending= True)
sns.barplot(data=data_Category_Name.head(10), x='Number of Orders',y = 'Category Name')


# In[28]:


df1.head()


# In[29]:


df2=df1.drop(['Product Status'], axis=1)


# In[30]:


df1


# In[31]:


from sklearn.preprocessing import LabelEncoder


# In[32]:


labelencoder = LabelEncoder()
X=df2
X['Type'] = labelencoder.fit_transform(X['Type'])
X['Delivery Status'] = labelencoder.fit_transform(X['Delivery Status'])
X['Customer Country'] = labelencoder.fit_transform(X['Customer Country'])
X['Customer Segment'] = labelencoder.fit_transform(X['Customer Segment'])
X['Customer State'] = labelencoder.fit_transform(X['Customer State'])
X['	Category Name'] = labelencoder.fit_transform(X['Category Name'])
X['Department Name'] = labelencoder.fit_transform(X['Department Name'])
X['Market'] = labelencoder.fit_transform(X['Market'])
X['Order City'] = labelencoder.fit_transform(X['Order City'])
X['Order Region'] = labelencoder.fit_transform(X['Order Region'])
X['Order State'] = labelencoder.fit_transform(X['Order State'])
X['Order Status'] = labelencoder.fit_transform(X['Order Status'])
X['Shipping Mode'] = labelencoder.fit_transform(X['Shipping Mode'])
X['Order Country'] = labelencoder.fit_transform(X['Order Country'])
X['Customer City'] = labelencoder.fit_transform(X['Customer City'])
X['Category Name'] = labelencoder.fit_transform(X['Category Name'])
X['Product Name'] = labelencoder.fit_transform(X['Product Name'])


# In[33]:


df3=df2


# In[34]:


df3


# In[35]:


df2.drop(['Category Name', 'Customer Id', 'Customer Zipcode','Order Customer Id','order date (DateOrders)','Order Id','Order Item Cardprod Id','Order Item Id','Product Card Id','Product Category Id','Product Image','Customer_ID_STR'], axis = 1, inplace = True)


# In[36]:


df2.describe()


# In[37]:


df2=df2.drop(['shipping date (DateOrders)'], axis=1)


# In[38]:


#normalization function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(df2)
b = df_norm.describe()
b


# In[39]:


from sklearn.preprocessing import MinMaxScaler
minmaxscale = MinMaxScaler()

new_data = minmaxscale.fit_transform(df2)
df_data = pd.DataFrame(new_data)
minmax_res = df_data.describe()


# In[40]:


minmax_res


# In[41]:


df_data


# In[42]:


df3


# In[43]:


df1.to_csv(r"C:\\Users\nikhil\Downloads\df1.csv")


# In[ ]:




