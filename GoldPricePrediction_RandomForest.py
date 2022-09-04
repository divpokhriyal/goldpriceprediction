#!/usr/bin/env python
# coding: utf-8

# In[36]:


#Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score


# In[37]:


#Loading the dataset into a pandas dataframe

gold_data = pd.read_csv('goldpricedata.csv')


# In[38]:


gold_data


# In[39]:


#exploring the dataset

gold_data.info()

#checking the number of missing vakues

gold_data.isnull().sum()

# -- no missing values -- 


# In[40]:


#statistical measures of this data

gold_data.describe()


# In[41]:


#Correlation between different columns -- positive & negative

correlation = gold_data.corr()


# In[42]:


#a heat map representing the correlation

plt.figure(figsize= (10, 10))
sns.heatmap(correlation , cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':8}, color = 'Red')


# In[43]:


#Correlation values of GLD

print(correlation['GLD'])


# In[44]:


#Distribution of gold prices 

sns.displot(gold_data['GLD'], color = 'pink')


# In[45]:


# splitting the features and target column 

X = gold_data.drop(['Date', 'GLD'], axis =1)
Y = gold_data['GLD']


# In[46]:


X


# In[47]:


Y


# In[48]:


#Slpitting into training and testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state =2)


# In[49]:


#Model Training - Random Forest Regressor

regressor = RandomForestRegressor(n_estimators=100)


# In[50]:


#Training the model

regressor.fit(X_train, Y_train)


# In[51]:


#Model evaluation - test data 

test_data_prediction = regressor.predict(X_test)

print(test_data_prediction)


# In[53]:


#R squared error

error_score = r2_score(Y_test, test_data_prediction)

print("R squared error = ", error_score)


# In[54]:


#Comparing the actual and predicted values 

Y_test = list(Y_test)


# In[71]:


plt.plot(Y_test, color = 'green', label ='Actual Value')
plt.plot(test_data_prediction, color = 'red', label ='Predicted Value')
plt.title = ('Actual Price vs Predicted Price')
plt.xlabel('Number of Values')
plt.ylabel('Gold Price')
plt.legend
plt.show()


# In[ ]:





# In[ ]:




