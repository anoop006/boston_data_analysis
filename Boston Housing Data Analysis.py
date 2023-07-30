#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import datetime
import scipy.stats

get_ipython().run_line_magic('matplotlib', 'inline')
#sets the default autosave frequency in seconds
get_ipython().run_line_magic('autosave', '60')
sns.set_style('dark')
sns.set(font_scale=1.2)

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows',None)
pd.set_option('display.width', 1000)

np.random.seed(0)
np.set_printoptions(suppress=True)


# In[4]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)


# In[6]:


boston_df


# In[9]:


boston_df.info


# In[10]:


boston_df.describe


# In[ ]:


Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT', 'MEDV'], dtype='object')


# In[13]:


boston_df.hist(bins=50, figsize=(20,10))
plt.suptitle('Feature Distribution', x=0.5, y=1.02, ha='center', fontsize='large')
plt.tight_layout()
plt.show()


# In[35]:


boston_df.loc[(boston_df["AGE"] <= 35),'age_group'] = '35 years and younger'
boston_df.loc[(boston_df["AGE"] > 35) & (boston_df["AGE"]<70),'age_group'] = 'between 35 and 70 years'
boston_df.loc[(boston_df["AGE"] >= 70),'age_group'] = '70 years and older'


# In[37]:


boston_df


# In[40]:


plt.figure(figsize=(10,5))
sns.boxplot(x=boston_df.MEDV, y=boston_df.age_group, data=boston_df)
plt.title("Boxplot for the MEDV variable vs the AGE variable")
plt.show()


# In[45]:


plt.figure(figsize=(10,5))
sns.scatterplot(x=boston_df.NOX, y=boston_df.INDUS, data=boston_df)
plt.title("Relationship between NOX and INDUS")
plt.show()


# In[46]:


plt.figure(figsize=(10,5))
sns.distplot(a=boston_df.PTRATIO,bins=10, kde=False)
plt.title("Histogram for the pupil to teacher ratio variable")
plt.show()


# In[47]:


boston_df


# In[ ]:


#hypothesis testing


# In[48]:


boston_df["CHAS"].value_counts()


# In[49]:


a = boston_df[boston_df["CHAS"] == 0]["MEDV"]
a


# In[50]:


b = boston_df[boston_df["CHAS"] == 1]["MEDV"]
b


# In[51]:


scipy.stats.ttest_ind(a,b,axis=0,equal_var=True)


# In[53]:


boston_df["AGE"].value_counts()


# In[55]:


boston_df.loc[(boston_df["AGE"] <= 35),'age_group'] = '35 years and younger'
boston_df.loc[(boston_df["AGE"] > 35) & (boston_df["AGE"]<70),'age_group'] = 'between 35 and 70 years'
boston_df.loc[(boston_df["AGE"] >= 70),'age_group'] = '70 years and older'


# In[56]:


boston_df


# In[57]:


low = boston_df[boston_df["age_group"] == '35 years and younger']["MEDV"]
mid = boston_df[boston_df["age_group"] == 'between 35 and 70 years']["MEDV"]
high = boston_df[boston_df["age_group"] == '70 years and older']["MEDV"]


# In[60]:


f_stats,p_value = scipy.stats.f_oneway(low,mid,high,axis=0)


# In[61]:


print("F-Statistic={0}, P-value={1}".format(f_stats,p_value))


# In[62]:


pearson,p_value = scipy.stats.pearsonr(boston_df["NOX"],boston_df["INDUS"])
print("Pearson Coefficient value={0}, P-value={1}".format(pearson,p_value))


# In[64]:


boston_df.columns


# In[66]:


y = boston_df['MEDV']
x = boston_df['DIS']


# In[67]:


x= sm.add_constant(x)


# In[68]:


results = sm.OLS(y,x).fit()


# In[ ]:


# results.summary()


# In[70]:


np.sqrt(0.062)


# In[71]:


The square root of R-squared is 0.25, which implies weak correlation between both features


# In[72]:


#correlation


# In[73]:


boston_df.corr()


# In[76]:


plt.figure(figsize=(16,9))
sns.heatmap(boston_df.corr(),cmap="coolwarm",annot=True,fmt='.2f',linewidths=2, cbar=False)
plt.show()


# In[ ]:




