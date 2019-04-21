'''
Created on Apr 7, 2019

@author: sycamore
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, forest
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from numpy import full



rooms_lx, bedrooms_lx, population_lx, household_lx = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X, y = None):
        return self
    def transform(self,X,y=None):
        rooms_per_household = X[:,rooms_lx]/X[:, household_lx]
        population_per_household = X[:,population_lx]/X[:, household_lx]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_lx]/X[:,rooms_lx]
            return np.c_[X, rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

    
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y = None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values

def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def display_scores(scores):
    print("scores:", scores)
    print("mean:",scores.mean())
    print("Standard deviation:", scores.std())
    print("*********************************************************")

orig_ca_hsg_data = pd.read_csv("c:\\kaggle-dataset\\ca_housing_1990.csv")

print(orig_ca_hsg_data.shape)
print(orig_ca_hsg_data.head(5))
print(orig_ca_hsg_data.info())
print(orig_ca_hsg_data["ocean_proximity"].value_counts())
print(orig_ca_hsg_data.describe())
#train_set, test_set = split_train_test(orig_ca_hsg_data, 0.2)
train_set, test_set = train_test_split(orig_ca_hsg_data,test_size=0.2,random_state=42)
print(len(train_set)," train +",len(test_set), " test")
orig_ca_hsg_data['income_cat']=np.ceil(orig_ca_hsg_data['median_income']/1.5)
orig_ca_hsg_data['income_cat'].where(orig_ca_hsg_data['income_cat'] < 5, 5.0, inplace=True)
#orig_ca_hsg_data.hist(bins=50, figsize=(20,15))
#pyplt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(orig_ca_hsg_data, orig_ca_hsg_data['income_cat']):
    strat_train_set = orig_ca_hsg_data.loc[train_index]
    strat_test_set = orig_ca_hsg_data.loc[test_index]

print(orig_ca_hsg_data['income_cat'].value_counts() / len(orig_ca_hsg_data))
for set_ in (strat_train_set,strat_test_set):
    set_.drop('income_cat',axis=1,inplace=True)
    
mod_ca_hsg_train_data = strat_train_set.copy()

mod_ca_hsg_train_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, 
                           s=mod_ca_hsg_train_data['population']/100, label='population',
                           figsize=(10,7), c='median_house_value',cmap=pyplt.get_cmap("jet"), colorbar=True
                           )
#pyplt.legend()
#pyplt.show()


mod_ca_hsg_train_data['rooms_per_household']=mod_ca_hsg_train_data['total_rooms']/mod_ca_hsg_train_data['households']
mod_ca_hsg_train_data['bedrooms_per_room']=mod_ca_hsg_train_data['total_bedrooms']/mod_ca_hsg_train_data['total_rooms']
mod_ca_hsg_train_data['population_per_household']=mod_ca_hsg_train_data['population']/mod_ca_hsg_train_data['households']

corr_matrix=mod_ca_hsg_train_data.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))
#mod_ca_hsg_train_data.plot(kind='scatter', x='median_income',y='median_house_value',alpha=0.1)
#pyplt.show()

housing=strat_train_set.drop('median_house_value',axis=1)
housing_labels=strat_train_set['median_house_value'].copy()

housing_cat=housing['ocean_proximity']
'''
encoder = LabelEncoder()

housing_cat_encoded=encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
print(encoder.classes_)
'''

encoder = LabelBinarizer(sparse_output=True)
housing_cat_1hot=encoder.fit_transform(housing_cat)
print(housing_cat_1hot)

imputer = SimpleImputer(strategy='median')
housing_num=housing.drop('ocean_proximity',axis=1)
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)
X=imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns=housing_num.columns)
print(housing_tr.info())
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']
#attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
#housing_extra_attribs=attr_adder.fit(housing.values)
num_pipeline = Pipeline([
        ('selector',DataFrameSelector(num_attribs)),
        ('imputer',SimpleImputer(strategy="median")),
        ('attribs_adder',CombinedAttributesAdder()),
        ('std_scaler',StandardScaler())
    ])
cat_pipeline = Pipeline([
    ('selector',DataFrameSelector(cat_attribs)),
    ('one_hot_encoder', OneHotEncoder(sparse=False))
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline',num_pipeline),
        ('cat_pipeline',cat_pipeline)
    ])



housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)

lin_reg_model = LinearRegression()
lin_reg_model.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:",lin_reg_model.predict(some_data_prepared))
print("Labels:",list(some_labels))

housing_predictions = lin_reg_model.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
print("Linear regression Root Mean Squared Error:",lin_rmse)
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

print("Decision Tree Regression Root Mean Squared Error:",tree_rmse)

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv = 10)
tree_rmse_scores = np.sqrt(-scores)
print("Scored for  Decision Tree Regression")
display_scores(tree_rmse_scores)


lin_scores = cross_val_score(lin_reg_model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv = 10)    #2
lin_rmse_scores = np.sqrt(-lin_scores)
print("Scored for  Linear Regression")
display_scores(lin_rmse_scores)

param_grid = [
        {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
        {'bootstrap': [False],'n_estimators': [3,10],'max_features':[2,3,4]}
    ]
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid,cv=5,scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
#forest_scores = cross_val_score(grid_search, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv = 10)    #1
#forest_rmse_scores = np.sqrt(-forest_scores)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score), params)
print("Scored for  Random Forest Regression")
#display_scores(forest_rmse_scores)

final_model = grid_search.best_estimator_
X_test = strat_test_set.drop('median_house_value',axis=1)
y_test = strat_test_set['median_house_value'].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("Final RMSE",final_rmse)

svr = svm.SVR()
svr.fit(housing_prepared, housing_labels)
svr_scores = cross_val_score(svr, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv = 10)
svr_rmse_scores = np.sqrt(-svr_scores)
print("Scored for  Support Vector Regression")
display_scores(svr_rmse_scores)

mlp =  MLPRegressor()
mlp.fit(housing_prepared, housing_labels)
mlp_scores = cross_val_score(mlp, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv = 10)
mlp_rmse_scores = np.sqrt(-mlp_scores)
print("Scored for  Multi-Layer Perceptron Regression")
display_scores(mlp_rmse_scores)
