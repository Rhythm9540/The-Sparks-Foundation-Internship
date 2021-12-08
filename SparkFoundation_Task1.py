#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation

# ## Task1: Predict the percentage of student based on number of study hours

# #### Dataset link :- https://bit.ly/w-data

# ### Author : Rhtyhm Kaushik
# 

# In[1]:


# Importing the required libraries
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv(r"D:\Data Science\Spark Foundation\DataSet/student_scores - student_scores.csv")


# ### Inspecting the dataframe
# 
# inspecting the dataframe for dimensions, null-values, and summary of different columns

# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# ### Data Visualization

# In[8]:


# This id to ignore warning
import warnings as wgs
wgs.filterwarnings("ignore")


# In[9]:


sns.scatterplot("Hours","Scores",color='r',data=df)
plt.title("Hours vs Scores")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.grid()
plt.show()


# In[21]:


sns.jointplot(x='Hours',y='Scores',data=df)


# #### Preparing the Data

# In[10]:


X = df.drop('Scores',axis=1)
y = df['Scores']


# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=0)


# In[12]:


print('X_train is', X_train.shape)
print('X_test is', X_test.shape)


# In[13]:


print('y_train is', y_train.shape)
print('y_test is', y_test.shape)


# In[ ]:





# #### Training the Algorithm

# In[14]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)


# In[15]:


print("The intercept term of the linear model",LR.intercept_)
print("The coefficient term of the linear model",LR.coef_)


# In[16]:


#plotting the line(y=mx+c)
line = LR.coef_ * X + LR.intercept_
plt.scatter(X,y)
plt.plot(X,line)
plt.show()


# In[ ]:





# In[17]:


test_pred = LR.predict(X_test)


# In[18]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': test_pred})  
df 


# In[19]:


#Testing with data
hours = 9.25
random_pred = LR.predict([[hours]])
print("The Predicted Score if person studies for",hours,'hours is ', random_pred[0])


# #### Evaluating the Model

# In[20]:


from sklearn import metrics
print("Mean Squared Error is ",metrics.mean_squared_error(y_test,test_pred))
print("Mean Absolute Error is ",metrics.mean_absolute_error(y_test,test_pred))


# In[ ]:





# In[ ]:




