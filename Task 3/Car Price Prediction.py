#!/usr/bin/env python
# coding: utf-8

# # Importing Libraies

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')


# # Data Loading and Manipulating

# In[2]:


Car = pd.read_csv("D:\Disk F\My Stuff\Internships\OASIS INFOBYTE\TASK 3\Real Dataset.csv")
Car


# In[3]:


Car.head()


# In[4]:


Car.info()


# In[5]:


Car.describe()


# In[6]:


Car.isnull()


# In[7]:


Car.isnull().sum()


# In[8]:


Car.duplicated()


# In[9]:


Car.duplicated().sum()


# # Data Pre-Processing

# In[53]:


le = LabelEncoder()


# In[11]:


Car['fueltype'].unique()


# In[12]:


Car['fueltype'] = Car['fueltype'].map({'gas':0, 'diesel':1})
Car['fueltype'].unique()


# In[ ]:





# In[13]:


Car['aspiration'].unique()


# In[14]:


Car['aspiration'] = Car['aspiration'].map({'std':0, 'turbo':1})
Car['aspiration'].unique()


# In[ ]:





# In[15]:


Car['doornumber'].unique()


# In[16]:


Car['doornumber'] = Car['doornumber'].map({'two':2, 'four':4})
Car['doornumber'].unique()


# In[ ]:





# In[17]:


Car['carbody'] = le.fit_transform(Car['carbody'])
Car.head()


# In[ ]:





# In[18]:


Car['drivewheel'].unique()


# In[19]:


Car['drivewheel']= Car['drivewheel'].map({'fwd':0, 'rwd':1, '4wd':4})
Car['drivewheel'].unique()


# In[ ]:





# In[20]:


Car['enginelocation'].unique()


# In[21]:


Car['enginelocation']= Car['enginelocation'].map({'front':0, 'rear':1})
Car['enginelocation'].unique()


# In[ ]:





# In[22]:


Car['enginetype'] = le.fit_transform(Car['enginetype'])
Car.head()


# In[ ]:





# In[23]:


Car['cylindernumber'] = le.fit_transform(Car['cylindernumber'])
Car.head()


# In[ ]:





# In[24]:


Car['fuelsystem'] = le.fit_transform(Car['fuelsystem'])
Car.head()


# # Machine learning

# In[25]:


from sklearn.model_selection import train_test_split

X = Car.drop(['CarName', 'price'], axis=1)
y = Car['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=3332)


# # 1) Linear Regression Model

# In[26]:


#linear regression model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()

LR.fit(X_train, y_train)


# In[27]:


#print matrix to get performance
Accuracy_1 = LR.score(X_test, y_test) * 100
print('Accuracy:', Accuracy_1)

prediction_1 = LR.predict(X_test)
print(prediction_1)

Actual_1 = y_test
print(Actual_1)


# # 2) XG Boost Regression Model

# In[35]:


from xgboost import XGBRegressor
XGB = XGBRegressor()

XGB.fit(X_train, y_train)


# In[39]:


#print matrix to get performance
Accuracy_2 = XGB.score(X_test, y_test) * 100
print('Accuracy:', Accuracy_2)

prediction_2 = XGB.predict(X_test)
print(prediction_2)

Actual_2 = y_test
print(Actual_2)


# # 3) Random Forest Regression Model

# In[41]:


from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()

RF.fit(X_train, y_train)


# In[42]:


#print matrix to get performance
Accuracy_3 = RF.score(X_test, y_test) * 100
print('Accuracy:', Accuracy_3)

prediction_3 = RF.predict(X_test)
print(prediction_3)

Actual_3 = y_test
print(Actual_3)


# # 4) Gradient Boost Regression Model

# In[43]:


from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor()

GBR.fit(X_train, y_train)


# In[44]:


#print matrix to get performance
Accuracy_4 = GBR.score(X_test, y_test) * 100
print('Accuracy:', Accuracy_4)

prediction_4 = GBR.predict(X_test)
print(prediction_4)

Actual_4 = y_test
print(Actual_4)


# # Results

# In[45]:


Result = pd.DataFrame({'Models':['LR', 'XGB', 'RF','GBR'],
             'Accuracy':[Accuracy_1, Accuracy_2, Accuracy_3, Accuracy_4]})

Result


# In[47]:


fig = plt.figure(figsize = (6, 6))
bars = plt.bar(Result['Models'], Result['Accuracy'], width = 0.6)
bars[0].set_color('orange')
bars[1].set_color('blue')
bars[2].set_color('green')
plt.xlabel('Models')
plt.ylabel('Accuracy in %')
plt.title('Performed models & their Aaccuracy')
plt.show()


# # Model Saving &Testing

# In[49]:


#Saving the Model

GBR = RandomForestRegressor()
GBR_Model = GBR.fit(X, y)

import joblib
joblib.dump(GBR_Model, 'Car_Price_Predictor')

Car_Model = joblib.load('Car_Price_Predictor')



Car1_Details = pd.DataFrame({
    'symboling':0,  
    'fueltype':1, 
    'aspiration':1, 
    'doornumber':4,
    'carbody':2, 
    'drivewheel':1, 
    'enginelocation':1, 
    'wheelbase':95, 
    'carlength':173,
    'carwidth':69, 
    'carheight':49, 
    'curbweight':2450, 
    'enginetype':0, 
    'cylindernumber':6,
    'enginesize':155, 
    'fuelsystem':2, 
    'boreratio':3.30, 
    'stroke':3.20, 
    'compressionratio':8.99,
    'horsepower':177, 
    'peakrpm':5500, 
    'citympg':14, 
    'highwaympg':17}, index=[0])


# In[50]:


print('Car Price is =', Car_Model.predict(Car1_Details))


# In[51]:


Car2_Details = pd.DataFrame({
    'symboling':-2,  
    'fueltype':0, 
    'aspiration':0, 
    'doornumber':2,
    'carbody':0, 
    'drivewheel':2, 
    'enginelocation':1, 
    'wheelbase':2470, 
    'carlength':4324,
    'carwidth':1864, 
    'carheight':1304, 
    'curbweight':1610, 
    'enginetype':0, 
    'cylindernumber':8,
    'enginesize':2998, 
    'fuelsystem':2, 
    'boreratio':3.30, 
    'stroke':3.20, 
    'compressionratio':8.99,
    'horsepower':335, 
    'peakrpm':6600, 
    'citympg':10, 
    'highwaympg':12}, index=[0])


# In[52]:


print('Car Price is =', Car_Model.predict(Car2_Details))


# In[ ]:





# In[ ]:




