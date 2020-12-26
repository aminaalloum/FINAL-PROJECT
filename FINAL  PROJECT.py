#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import pandas as pd
data=pd.read_csv("application_record.csv.zip",engine="python",sep=",")
data


# # data preprocessing

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


import pandas as pd
df=pd.read_csv("application_record.csv.zip",engine="python",sep=",")
df


# In[5]:



df.isnull().sum()


# In[6]:


df["OCCUPATION_TYPE"].value_counts()


# In[7]:


df["OCCUPATION_TYPE"].fillna("Laborers")


# In[ ]:





# In[8]:


df.isnull().sum()


# In[9]:


duplicateRows = df[df.duplicated(keep='first')]
print("Duplicate Rows except last occurrence based on all columns are :")
print(duplicateRows)


# In[10]:


df=df.drop_duplicates()
df


# In[ ]:





# In[ ]:





# In[11]:


credit_df=pd.read_csv("credit_record.csv.zip",engine="python",sep=",")
credit_df
credit_df.STATUS.replace('X', 0, inplace=True)
credit_df.STATUS.replace('C', 0, inplace=True)
credit_df.STATUS = credit_df.STATUS.astype('int')
credit_df


# In[12]:


import pandas as pd
df=pd.read_csv("application_record.csv.zip",engine="python",sep=",",encoding = "utf-8-sig") 
df=pd.merge(df,credit_df)
df.head()


# In[13]:


df.isnull().sum()


# In[14]:


df.drop_duplicates(inplace=True)
df


# In[15]:


df["OCCUPATION_TYPE"].value_counts()


# In[16]:


df=df.fillna("Laborers")


# In[17]:


df.isnull().sum()


# #Data Visualization

# In[18]:


ax=plt.subplots(figsize=(20,8))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[38]:



fig, ax = plt.subplots()
ax.stackplot( df["DAYS_BIRTH"],df["CODE_GENDER"],
             labels=df.OCCUPATION_TYPE.keys(),color="blue")


# In[21]:


sns.countplot(x=df["CODE_GENDER"])


# In[78]:




binary_features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']
binary_df = df[binary_features+['STATUS']].replace('M', 1).replace('F', 0).replace('Y', 1).replace('N', 0)
dict_list = []
for feature in binary_features:
    for one_type in [0, 1]:
        dict_list.append({'feature': feature,
                          'type': one_type,
                          'reject_rate in type': len(binary_df[binary_df[feature]==one_type][binary_df.STATUS==1])/len(binary_df[binary_df[feature]==one_type]),
                          'count': len(binary_df[binary_df[feature]==one_type]),
                          'Reject_count': len(binary_df[binary_df[feature]==one_type][binary_df.STATUS==1])
                         })

group_binary = pd.DataFrame.from_dict(dict_list)
sns.barplot(x="feature", y="reject_rate in type", hue="type", data=group_binary)
plt.show()
group_binary


# In[44]:


pd.crosstab(df.CNT_CHILDREN,df.STATUS).plot(kind='bar', figsize=(12,6))


# In[47]:


df['OCCUPATION_TYPE'].value_counts().sort_values().plot(kind='barh', figsize=(9,12), alpha=0.7,color="pink")


# In[24]:


pd.crosstab(df.CODE_GENDER, df.NAME_FAMILY_STATUS).plot(kind='bar', figsize=(9,6))


# In[25]:


sns.lmplot(data=df, x='DAYS_BIRTH', y='AMT_INCOME_TOTAL')


# In[49]:


df['NAME_FAMILY_STATUS'].value_counts().sort_values().plot(kind="barh",color="purple")


# In[50]:


df["CNT_CHILDREN"].value_counts().sort_values().plot(kind="barh",color="brown")


# In[64]:


gen_per = df.CODE_GENDER.value_counts(normalize = True)
gen_per
gen_per.plot.pie()
plt.show()


# In[54]:


df["OCCUPATION_TYPE"].value_counts().sort_values().plot(kind="barh",color="red")


# In[58]:


pd.crosstab(df.NAME_EDUCATION_TYPE, df.STATUS).plot(kind='barh', figsize=(9,9))


# In[80]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder() 
df['CODE_GENDER']=label_encoder.fit_transform(df["CODE_GENDER"])
df


# In[ ]:





# In[31]:


df


# In[ ]:





# In[72]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 
lgreg=LinearRegression ()
y=df["STATUS"].values.reshape(-1,1)
x=df["AMT_INCOME_TOTAL"].values.reshape(-1,1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)
print(x.shape)
print(y.shape)


# In[33]:


y_train


# In[34]:


lgreg.fit(x_train,y_train)
y_pred=lgreg.predict(x_test)


# In[35]:


print("Accruacy={:.2f}".format(lgreg.score(x_test,y_test)))


# In[36]:


plt.plot(x_train,lgreg.predict(x_train),color="blue")
plt.scatter(x_train,y_train,color="red")


# In[37]:


from sklearn.metrics import mean_squared_error, r2_score
mse= mean_squared_error(y_train,lgreg.predict(x_train))
mse


# In[38]:


r2 = r2_score(y_train,lgreg.predict(x_train))
r2


# In[76]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder() 
df['NAME_INCOME_TYPE']=label_encoder.fit_transform(df["NAME_INCOME_TYPE"])
df['NAME_EDUCATION_TYPE']=label_encoder.fit_transform(df["NAME_EDUCATION_TYPE"])
df["NAME_FAMILY_STATUS"]=label_encoder.fit_transform(df["NAME_FAMILY_STATUS"])
df["OCCUPATION_TYPE"]=label_encoder.fit_transform(df['OCCUPATION_TYPE'])
df["FLAG_OWN_CAR"]=label_encoder.fit_transform(df["FLAG_OWN_CAR"])
df["FLAG_OWN_REALTY"]=label_encoder.fit_transform(df["FLAG_OWN_REALTY"])
df["NAME_HOUSING_TYPE"]=label_encoder.fit_transform(df["NAME_HOUSING_TYPE"])
df


# In[81]:


x=df[["CODE_GENDER","AMT_INCOME_TOTAL"]]
y=df[["STATUS"]].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
lgreg.fit(x_train,y_train)
pred=lgreg.predict(x_test)


# In[ ]:





# In[40]:


plt.plot(x_test,pred,color="blue")
#plt.scatter(x_test,y_test,color="red")


# In[82]:


x=df["OCCUPATION_TYPE"].values.reshape(-1,1)
y=df["NAME_INCOME_TYPE"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
lgreg.fit(x_train,y_train)
pred=lgreg.predict(x_test)
from sklearn.preprocessing import PolynomialFeatures
lg=LinearRegression()
poly=PolynomialFeatures(degree=2)


# In[83]:


x_=poly.fit_transform(x)
lg.fit(x_,y)


# In[84]:


x_train_poly=poly.fit_transform(x_train)
lg.fit(x_train_poly,y_train)
x_test_fit=poly.fit_transform(x_test)
lg.predict(x_test_fit)
y_pred=lg.predict(x_test_fit)


# In[45]:


import matplotlib.pyplot as plt
plt.scatter(x,y,color="red")
plt.plot(x,lg.predict(poly.fit_transform(x)),color="blue")


# In[46]:



mse=mean_squared_error(y_test,y_pred)
mse


# In[47]:


r2=r2_score(y_test,y_pred)
r2


# In[85]:


from sklearn.linear_model import LogisticRegression
x=df["NAME_EDUCATION_TYPE"].values.reshape(-1,1)
y=df["STATUS"].values
logreg=LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
logreg.fit(x_train,y_train)
y_pred=lgreg.predict(x_test)


# In[ ]:


import seaborn as sns
sns.regplot(x=x_test,y=y_test,data=df,logistic=True)


# In[ ]:


print("Accruacy={:.2f}".format(logreg.score(x_test,y_test)))


# In[ ]:


confusion_matrix=pd.crosstab(y_test,y_pred,rownames=["actual"],colnames=["predicted"])
print(confusion_matrix)


# In[86]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))


# In[ ]:


import numpy as np
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 8):
    knn = KNeighborsClassifier(n_neighbors=i,metric='euclidean')
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)


# In[ ]:


from sklearn import tree 
clf=tree.DecisionTreeClassifier()
clf=clf.fit(x_train,y_train)


# In[ ]:


import graphviz
dotfile = open("dtree.pdf", 'w')
ts=tree.export_graphviz(clf, out_file = dotfile, feature_names=x.columns, filled=True, 
                    rounded=True, impurity=False, class_names=["Survived","no survived"])
graph=graphviz.Source(dotfile)
graph.render("dtree.pdf",view=True)


# In[60]:


y_pred=clf.predict(x_test)
from sklearn import metrics 
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# In[61]:


from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators = 100)
regressor.fit(x,y)


# In[62]:


y_pred=clf.predict(x_test)
print("Accruacy:",metrics.accuracy_score(y_test,y_pred))


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
model=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
clut_labels=model.fit_predict(df.head(5000))
clut_labels


# In[ ]:


agglom=pd.DataFrame(clut_labels)
agglom.head(100)


# In[81]:


from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
dend2=shc.dendrogram(shc.linkage(df.head(300),method="complete"))


# In[83]:


from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4,random_state=0)
kmeans.fit(df)


# In[85]:


labels=pd.DataFrame(kmeans.labels_)
labels.head()


# In[86]:


import numpy as  np
k=range(1,16)
sosd=[]
for k in k:
    km = KMeans(n_clusters=k, init='k-means++', random_state= 0)  
    km=km.fit(df)  
    sosd.append(kmeans.inertia_)


# In[ ]:


frequent_itm=apriori(df,min_support=0.6,use_colnames=True)
frequent_itm


# In[ ]:


from mlxtend.frequent_patterns import association_rules
association_rules(frequent_itm,metric="confidence",min_threshold=0.7)


# In[ ]:


association_rules(frequent_itm,metric="lift",min_threshold=1.20)

