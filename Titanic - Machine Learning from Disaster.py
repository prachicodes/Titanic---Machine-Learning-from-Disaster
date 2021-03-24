#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[63]:


df = pd.read_csv(r"C:\Users\prach\Desktop\Data Science\ML projects\Titnic_train.csv")


# Variable info:
# 
#         PassengerId: Unique id number to each passenger.
#         Survived: Passenger survive(1) or died(0).
#         Pclass: Passenger class.
#         Name: Name of passenger.
#         Sex: Sex of passenger.
#         Age: Age of passenger.
#         SibSp: Number of siblings and spouses.
#         Parch: Number of parents/childrens.
#         Ticket: Ticket number.
#         Fare: amount of money spent on ticket
#         Cabin: Cabin category
#         Embarked: port where passenger embarked(S=Southampton,C=Cherbourg,Q=Queenstown)

# In[64]:


df.head()


# In[65]:


df.info()


# In[6]:


#There are null value in Age, Cabin and Embarked column 


# In[66]:


print(df.shape)
print(df.isnull().sum())


# In[67]:


print("The relation between sex of the person and if they survived or not?")
#select the female column who survived 
women = df.loc[df.Sex == 'female'] ['Survived']
rate_of_women = sum(women)/len(women)


#select the male column who survived 
male = df.loc[df.Sex == 'male'] ['Survived']
rate_of_male = sum(male)/len(male)

print("The percent of women who survived is:",(round(rate_of_women,2)*100),"%")
print("The percent of men who survived is:",(round(rate_of_male,2)*100),"%")


# Bar plot to indicate the number of people who survived

# In[68]:


import matplotlib.pyplot as plt

df['Survived'].value_counts().plot.bar()
plt.legend(['Survived Column'])
plt.xlabel("0-> Not survived | 1-> survived")
plt.show()


# Bar plot to indicate the count of poeple in each Parch

# In[69]:


import seaborn as sb
sb.countplot(x='Pclass',data=df, palette='hls')


# # Corelation between our target feature Survived and other features

# In[70]:


pearson_corr_train = df.corr(method = "pearson")
spearman_corr_train = df.corr(method = "spearman")
kendall_corr_train = df.corr(method = "kendall")

print(pearson_corr_train["Survived"])


# # Bar Plot for counting the frequency for each feature

# In[71]:


def bar_plot(variable):
    var=df[variable]
    var_value=var.value_counts()
    plt.figure(figsize=(9,3))
    plt.bar(var_value.index,var_value)
    plt.xticks(var_value.index,var_value.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n{}:".format(variable,var_value))
    
    
category= ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]
for i in category:
    bar_plot(i)


# # Co-relation between Survived and PClass

# In[53]:


# visualize the correlation between SURVIVED and Pclass using a scatter plot
print("The relation between Survived and Pclass:")
df[['Survived', 'Pclass']].groupby(['Pclass'],as_index=False).mean().plot.scatter('Survived','Pclass')
plt.show()


# From the above plot we can infer that higher the Pclass, the number of peole who survived is larger. 

# # Co-relation between SibSp and Survived

# In[72]:


# visualize the correlation between SURVIVED and Age using a scatter plot
df[['Survived', 'SibSp']].groupby(['SibSp'],as_index=False).mean().plot.scatter('Survived','SibSp')
plt.show()


# From the above graph, we can infer that higher sibsp less chances of survival. Thus, can be negatively co-related.

# # Handling Missing Values

# In[73]:


df.head()


# In[74]:


E = {'Embarked':{'C': 0, 'Q': 1, 'S': 2},'Sex': {'male':0,'female':1}}
df.replace(E, inplace = True)


# In[75]:


df.head()


# # Feature selection and Missing Values

# In[76]:


#to drop values 
train_df = df.drop(['Name','PassengerId','Ticket','Cabin'],1)
train_df .head()


# We dropped the columns of Name, PassengerId, Ticket, Cabin as they are no corelated to Survved column and does not provide any insights. 

# In[77]:


#train_df null values
train_df.isnull().sum()


# In[78]:


# visualize the correlation between Pclass and Age
grid = sns.FacetGrid(train_df, col='Pclass', size=3, aspect=0.8, sharey=False)
grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))
plt.show()


# In[79]:


#Filling in Age value after noticing the relation with Pclass

def age_approx1(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    
    
train_df['Age'] = train_df[['Age', 'Pclass']].apply(age_approx1, axis=1)


# In[83]:


#Filling in 2 missing values with Embraked = 2, as 2 is the mode
train_df['Embarked'].fillna(2, inplace = True)


# In[84]:


#train_df null values
train_df.isnull().sum()


# # Modeling 

# In[97]:


from sklearn.linear_model import LogisticRegression
logres = LogisticRegression()
x_train = train_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y_train = train_df['Survived']

logres.fit(x_train,y_train)

reg=logres.fit(x_train,y_train)


# In[98]:


test_df = pd.read_csv(r"C:\Users\prach\Desktop\Data Science\ML projects\Titanic_test.csv")
newemb = {'Embarked':{'C': 0, 'Q': 1, 'S': 2},'Sex': {'male':0,'female':1}}
test_df.replace(newemb, inplace = True)

#to drop values 
test_df = test_df.drop(['Name','PassengerId','Ticket','Cabin'],1)
test_df.head()

print(test_df.isnull().sum())

def age_approx12(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    
    
test_df['Age'] = test_df[['Age', 'Pclass']].apply(age_approx12, axis=1)

test_df['Fare'].fillna(0, inplace = True)


# In[100]:


predictnow=reg.predict(test_df)
print(predictnow)


# In[104]:


y_pred_log_reg = logres.predict(test_df)
acc_log_reg = round( logres.score(x_train, y_train) * 100, 2)
print ('The accuracy of Logistic Regression model',str(acc_log_reg) + ' percent')

