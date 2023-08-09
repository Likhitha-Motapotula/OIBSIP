#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classification

# In[24]:


#An iris dataset has 3 types of flowers i.e, setosa, virginica, versicolor 
#with four main characteristics sepal length, sepal width, petal length , petal width
#loading the data

import numpy as np
import matplotlib.pyplot as plt #used for data visualization
import seaborn as sns #data visualization
import pandas as pd #used for loading data from different sources
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[27]:


#cols = ['sepal_length','sepal_widht','petal_length','petal_width']
df = pd.read_csv("F:\data set\Iris.csv")
df.head()


# # # Analyze and visualize the dataset
# 

# In[28]:


df.describe()


# In[29]:


df


# In[30]:


df.info()


# In[31]:


df.shape


# In[32]:


if 'species' in df:
    unique_species = df.species.unique()
    print(unique_species)
else:
    print("")


# In[20]:


df.columns


# In[21]:


df.corr()


# In[34]:


df.groupby('Species').mean()


# # # Data Visualization
# 

# In[36]:


sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=df)
plt.show()


# In[38]:


sns.lineplot(data=df.drop(['Species'], axis=1))
plt.show()


# In[53]:


df.plot.hist(subplots=True, layout=(10,10), figsize=(45, 45), bins=20)
plt.show()


# In[54]:


sns.heatmap(df.corr(), annot=True)
plt.show()


# In[56]:


g = sns.FacetGrid(df, col='Species')
g = g.map(sns.kdeplot, 'SepalLengthCm')


# In[58]:


sns.pairplot(df)


# In[59]:


df.hist(color= 'mediumpurple' ,edgecolor='black',figsize=(10,10))
plt.show()


# In[60]:


df.corr().style.background_gradient(cmap='coolwarm').set_precision(2)


# # Machine Learning

# In[61]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[62]:


x = df.drop('Species', axis=1)
y= df.Species

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=5)


# ## K Neighbors Classifier

# In[63]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(x_train, y_train)

knn.score(x_test, y_test)


# ## Logistic Regression

# In[64]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x, y)
y_pred = logreg.predict(x)
print(metrics.accuracy_score(y, y_pred))


# ## Support Vector Machine

# In[65]:


from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(x_train, y_train)

svm.score(x_test, y_test)


# ## Decision Tree Classifier

# In[66]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)

dtree.score(x_test, y_test)


# In[ ]:




