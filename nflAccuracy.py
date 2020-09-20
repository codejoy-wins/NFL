#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rb_data = pd.read_csv('nfl2.csv')
X = rb_data.drop(columns=['fantasy'])
y = rb_data['fantasy']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
score


# In[14]:


dobbins = model.predict([[38,7]])
dobbins


# In[19]:


nick = model.predict([[20,17]])
nick


# In[ ]:




