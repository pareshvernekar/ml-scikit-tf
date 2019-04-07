import matplotlib.pyplot as pyplt
import numpy as np
import pandas as pd
import sklearn
import sklearn.svm
import sklearn.neighbors
import sklearn.linear_model


''

print("Project to determine life satisfaction index based on GDP")

bliDF = pd.read_csv("C:\kaggle-dataset\OECDBLI2017cleanedcsv.csv",encoding = "ISO-8859-1")
gdpDF = pd.read_csv("C:\kaggle-dataset\gdp_per_capita.csv",encoding = "ISO-8859-1")
print(bliDF.columns)
print(gdpDF.columns)
'''bliDF.plot(kind='bar',x='Country',y='Life satisfaction as avg score')
pyplt.show()

gdpDF.plot(kind='bar',x='Country',y='2015')
pyplt.show()
'''
mod_bliDF = bliDF[['Country','Life satisfaction as avg score']]
mod_gdpDF= gdpDF[['Country','2015']]
'''
mod_bliDF.plot(kind='bar',x='Country',y='Life satisfaction as avg score')
pyplt.show()

mod_gdpDF.plot(kind='bar',x='Country',y='2015')
pyplt.show()

print(mod_bliDF.head(20))
print(mod_gdpDF.head(20))
comb_DF = pd.concat([mod_bliDF,mod_gdpDF])
print(comb_DF.columns)
print(comb_DF.head)'''
mergeDF = pd.merge(mod_bliDF,mod_gdpDF,how='outer')
mergeDF=mergeDF.dropna(subset=['2015','Life satisfaction as avg score'])
print(mergeDF.head)
print(mergeDF.describe())
X=np.c_[mergeDF['2015']]
Y=np.c_[mergeDF['Life satisfaction as avg score']]
#model = sklearn.linear_model.LinearRegression()
model=sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
model.fit(X,Y)
X_new=[[22587]]
print(model.predict(X_new))