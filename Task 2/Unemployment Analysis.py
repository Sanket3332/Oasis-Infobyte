#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np 

import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
sns.set()
plt.rcParams["figure.figsize"] = (6,4)

import plotly.express as px
import plotly.graph_objects as pgo

import datetime as dt
import calendar

import warnings
warnings.filterwarnings('ignore')


# # Data Reading & Manipulating

# In[2]:


India_data = pd.read_csv("D:\Disk F\My Stuff\Internships\OASIS INFOBYTE\TASK 2\India.csv")
India_data


# In[3]:


India_data.columns = ['State', 'Date', 'Frequency', 'Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate', 'Area']


# In[4]:


India_data.head()


# In[5]:


India_data.info()


# In[6]:


India_data.describe()


# In[7]:


India_data.isnull()


# In[8]:


India_data.isna().sum()


# In[9]:


India_data.dropna(inplace=True)


# In[10]:


India_data.isnull()


# In[11]:


India_data.duplicated()


# In[12]:


India_data.duplicated().sum()


# # Data Pre-Processing

# In[14]:


India_data.dropna(axis = 0, thresh=1, inplace=True)


# In[15]:


India_data['Date'] = pd.to_datetime(India_data['Date'], dayfirst=True)

India_data['Month_Count'] = India_data['Date'].dt.month

India_data['Month'] = India_data['Month_Count'].apply(lambda x: calendar.month_abbr[x])
India_data.tail()


# In[80]:


msno.matrix(India_data)


# In[81]:


India_data['State'].value_counts()


# In[17]:


India_data.Month.value_counts()


# # State Wise Analysis

# In[82]:


India_1 = India_data[['State', 'Estimated Labour Participation Rate']].groupby('State').mean().sort_values('Estimated Labour Participation Rate', ascending=False)
India_1


# In[83]:


India_1[0:].plot(kind='bar', figsize=(12,6), color='olivedrab', width=0.5)
plt.xlabel('States')
plt.ylabel('Labour Participation Rate in %')
plt.title('Labour Participation Rate in different states of India.')
plt.show();


# In[84]:


India_2 = India_data[['State', 'Estimated Employed']].groupby('State').sum().sort_values('Estimated Employed', ascending=False)
India_2


# In[85]:


India_2[0:].plot(kind='bar', figsize=(9,5), color='slateblue', width=0.5)
plt.xlabel('States')
plt.ylabel("People Employed in 'MM'")
plt.title('State wise Employment of people in India.')
plt.show();


# In[86]:


India = India_data[['State', 'Estimated Unemployment Rate']].groupby('State').mean().sort_values('Estimated Unemployment Rate', ascending=False)
India


# In[87]:


India[0:].plot(kind='bar', figsize=(12,6), color='peru', width=0.5)
plt.ylabel('Unemployment Rate in %')
plt.xlabel('States')
plt.title('Unemployment Rate in different states of India.')
plt.show();


# In[88]:


State = India_data['State']

Labour_Participation_Rate = India_data['Estimated Labour Participation Rate']/100
Employment = India_data['Estimated Employed']/100000000
Unemployment_Rate = India_data['Estimated Unemployment Rate']/100

fig = pgo.Figure()
fig.add_trace(pgo.Bar(x = State, y = Labour_Participation_Rate, name = 'Labour Participation Rate', width = 0.8))
fig.add_trace(pgo.Bar(x = State, y = Employment, name = 'Employment Rate', width = 0.6))
fig.add_trace(pgo.Bar(x = State, y = Unemployment_Rate, name = 'Unemployment Rate', width = 0.4))


fig.update_layout(title = 'Labour Participation Vs Employment Rate Vs Unemployment Rate in particular State.')

fig.show()


# # Month Wise Analysis

# In[89]:


India_4 = India_data[['Month', 'Estimated Labour Participation Rate']].groupby('Month').mean().sort_values('Estimated Labour Participation Rate', ascending=True)
India_4


# In[90]:


India_4[0:].plot(kind='bar', figsize=(9,5), color='lightcoral', width=0.5)
plt.xlabel('Months')
plt.ylabel('Labour Participation Rate in %')
plt.title('Labour Participation Rate on Monthly basis in India.')
plt.show();


# In[91]:


India_5 = India_data[['Month', 'Estimated Employed']].groupby('Month').sum().sort_values('Estimated Employed', ascending=True)
India_5


# In[92]:


India_5[0:].plot(kind='bar', figsize=(9,5), color='lawngreen', width=0.5)
plt.xlabel('Months')
plt.ylabel('People Employed')
plt.title('Month wise Employment of people in India.')
plt.show();


# In[93]:


India_3 = India_data[['Month', 'Estimated Unemployment Rate']].groupby('Month').mean().sort_values('Estimated Unemployment Rate', ascending=True)
India_3


# In[94]:


India_3[0:].plot(kind='bar', figsize=(9,5), color='cadetblue', width=0.5)
plt.xlabel('Months')
plt.ylabel('Unemployment Rate in %')
plt.title('Months wise Unemployment Rate in India.')
plt.show();


# In[95]:


Month = India_data['Month']

Labour_Participation_Rate = India_data['Estimated Labour Participation Rate']/100
Employment = India_data['Estimated Employed']/100000000
Unemployment_Rate = India_data['Estimated Unemployment Rate']/100

fig = pgo.Figure()
fig.add_trace(pgo.Bar(x = Month, y = Labour_Participation_Rate, name = 'Labour Participation Rate', width = 0.7))
fig.add_trace(pgo.Bar(x = Month, y = Employment, name = 'Employment Rate', width = 0.5))
fig.add_trace(pgo.Bar(x = Month, y = Unemployment_Rate, name = 'Unemployment Rate', width = 0.3))


fig.update_layout(xaxis = {'categoryorder' : 'array', 'categoryarray' : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']}, 
                  title = 'Labour Participation Vs Employment Rate vS Unemployment Rate in Specific Month.')

fig.show()


# ## Yearly Analysis

# In[96]:


fig = px.bar(India_data, x ='State', y = 'Estimated Labour Participation Rate', color = 'State', animation_frame = 'Month', 
             title = 'Labour Participation Rate in particular States over an Year (May - Apr).')

fig.update_layout(xaxis = {'categoryorder' : 'total ascending'})

fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 3500

fig.show() 


# In[97]:


fig = px.bar(India_data, x ='State', y = 'Estimated Employed', color = 'State', animation_frame = 'Month', 
             title = 'Count of Employed Peoples in particular States over an Year (May - Apr).')

fig.update_layout(xaxis = {'categoryorder' : 'total ascending'})

fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 3500

fig.show() 


# In[99]:


fig = px.bar(India_data, x ='State', y = 'Estimated Unemployment Rate', color = 'State',  animation_frame = 'Month',
             title = 'Unemployment Rate in particular States over an Year (May - Apr).')

fig.update_layout(xaxis = {'categoryorder' : 'total ascending'})

fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 4000, 

fig.show() 


# # Region Wise Analysis

# In[36]:


plt.pie(India_data['Area'].value_counts(), explode = (0.05, 0), labels = ('Urban', 'Rural'), colors = ('blue', 'coral'), autopct = '%2.2f%%', shadow = True, startangle = -1)
plt.title(" Population of Areas")
plt.show()


# In[100]:


fig = px.bar(India_data, x ='State', y = 'Estimated Labour Participation Rate', animation_frame = 'Area', color = 'State',
             title = 'Labour Participation Rate in Rural & Urban areas of Different States in India.')

fig.update_layout(xaxis = {'categoryorder' : 'total descending'})

fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 4000

fig.show() 


# In[101]:


fig = px.bar(India_data, x ='State', y = 'Estimated Employed', animation_frame = 'Area', color = 'State',
             title = 'Count of Employed Peoples from Rural & Urban areas in different States of India.')

fig.update_layout(xaxis = {'categoryorder' : 'total ascending'})

fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 4000

fig.show() 


# In[102]:


fig = px.bar(India_data, x ='State', y = 'Estimated Unemployment Rate', color = 'State',  animation_frame = 'Area',
             title = 'Unemployment Rate in Rural & Urban areas of Different States in India.')

fig.update_layout(xaxis = {'categoryorder' : 'total descending'})

fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 4000

fig.show() 


# # Result

# In[103]:


Report = India_data.groupby(['Area', 'State'])['Estimated Unemployment Rate'].mean().reset_index()
Report.head()


# In[104]:


fig = px.sunburst(Report, path=['Area', 'State'], values = 'Estimated Unemployment Rate',
                  title = 'Multilevel Pie-Chart of Unemployment Rate in different Areas of Indian States.', height = 750)

fig.show()

