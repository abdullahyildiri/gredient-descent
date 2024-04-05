#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random as rd
import pandas as pd
import numpy as np


# In[2]:


data = {'year': [2018, 2015, 2020, 2012],
        'km': [50000, 80000, 20000, 120000],
        'engine_size':[1.6, 2.0, 1.8, 1.4],
        'brand': ['Toyota', 'Honda', 'Volkswagen', 'Ford'],
        'price': [120000, 90000, 150000, 60000]}


# In[3]:


type(data)


# In[4]:


df = pd.DataFrame(data)


# In[5]:


df


# In[6]:


from sklearn.preprocessing import LabelEncoder


# In[7]:


label_encoder=LabelEncoder()
df["brand_encoded"]=label_encoder.fit_transform(df["brand"])
df


# In[8]:


X_train = df[["year","km","engine_size","brand_encoded"]]


# In[9]:


y_train = df["price"]


# In[10]:


y_train


# In[11]:


type(y_train)
type(X_train)


# In[12]:


X_train.iloc[1][2]


# In[13]:


y_train


# In[14]:


w_init = pd.Series(rd.random() for _ in range(4))


# In[15]:


w_init


# In[16]:


b_init = 0.


# In[17]:


w_init.shape


# In[18]:


X_train.iloc[1].shape


# In[19]:


y_train.shape


# In[20]:


def predict(x, w, b):
    return np.dot(x, w)+b


# In[21]:


predict(X_train.iloc[0], w_init, b_init)


# In[22]:


y_train


# In[23]:


def compute_cost(X_train, y_train, w, b):
    cost=0.
    m=X_train.shape[0]
    for i in range(m):
        cost+=(predict(X_train.iloc[i], w, b) - y_train.iloc[i])**2
    return cost/2*m
    


# In[24]:


print(compute_cost(X_train, y_train, w_init, b_init))


# In[25]:


def gradient_descent(X_train, y_train, w, b, learning_rate, epochs):
    m = X_train.shape[0]
    n = X_train.shape[1]
    cost_history = []
    
    for epoch in range(epochs):
        dw = np.zeros(n)
        db = 0
        
        for i in range(m):
            y_pred = predict(X_train.iloc[i], w, b)
            dw += (y_pred - y_train.iloc[i]) * X_train.iloc[i]
            db += (y_pred - y_train.iloc[i])
        
        dw /= m
        db /= m
        
        w -= learning_rate * dw
        b -= learning_rate * db
        
        cost = compute_cost(X_train, y_train, w, b)
        cost_history.append(cost)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost}")
    
    return w, b, cost_history

# Başlangıç ağırlıkları ve bias
w_init = np.array([rd.random() for _ in range(4)])
b_init = 0.

# Hyperparameters
learning_rate = 0.0001
epochs = 1000

# Gradient descent ile modeli eğitme
w_trained, b_trained, cost_history = gradient_descent(X_train, y_train, w_init, b_init, learning_rate, epochs)


# In[ ]:





# In[ ]:





# In[ ]:




