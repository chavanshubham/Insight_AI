#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pickle 

from os.path import expanduser
from os.path import join


from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import fbeta_score

from scipy import stats



# In[2]:


df = pd.read_csv('ricedata.csv')
df


# In[3]:


df.isnull().sum()


# In[4]:


df = df.drop(['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1','SOIL TYPE PERCENT (Percent)','JANUARY RAINFALL VARIATION','FEBRUARY RAINFALL VARIATION','MARCH RAINFALL VARIATION','APRIL RAINFALL VARIATION','DECEMBER RAINFALL VARIATION'], axis=1)
df


# In[6]:


df.isnull().sum()





from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype is object:
        df[column_name] = le.fit_transform(df[column_name].split('\n'))
        x_array = np.array(column_name)
        normalized_X = preprocessing.normalize([x_array])
    else:
        pass


# In[10]:


df=df.replace(-1,0)
df=df.replace(np.nan,0)


# In[11]:


df=df.replace('',0)
for i in df.columns:
    if i=='':
        df.drop([i],axis=1,inplace=True)





# In[22]:


from sklearn.preprocessing import LabelEncoder
#
# Instantiate LabelEncoder
#
le = LabelEncoder()
#
# Encode single column status
#
df['State_Name'] = le.fit_transform(df['State Name'])
df['Dist_Name'] = le.fit_transform(df['Dist Name'])
df['Dist_Name']
# Print df.head for checking the transformation
#
df.head()


# In[40]:


x = df.iloc[:,[0,1,2,5,6,7]]
print(x)
y = df['RICE YIELD (Kg per ha)']
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[39]:


lr_clf = LogisticRegression(random_state=0, solver='lbfgs',
                          multi_class='multinomial').fit(X_train, y_train)

rf_clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, y_train)


# In[41]:


rf_clf


# In[47]:


# Actual class predictions
rf_predictions = rf_clf.predict(X_test)
print(rf_predictions)
# Probabilities for each class
rf_probs = rf_clf.predict_proba(X_test)[:, 1]
print(rf_probs)


# In[49]:


lr_predictions = lr_clf.predict(X_test)
# Probabilities for each class
print(lr_predictions)
lr_probs = lr_clf.predict_proba(X_test)[:, 1]
print(lr_probs)


# In[52]:


errors = abs(rf_predictions - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[53]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:





# In[ ]:


# Saving model to disk
pickle.dump(rf_clf, open('crop.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('crop.pkl','rb'))
print(model.predict([[5,2000,14,19.75]]))




