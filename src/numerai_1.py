import sys

import numerapi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingRegressor
from sklearn.decomposition import PCA

#Upload data - note can download directly from python API
train_data=pd.read_csv('../data/numerai_dataset_252/numerai_training_data.csv').set_index('id')
test_data=pd.read_csv('../data/numerai_dataset_252/numerai_tournament_data.csv').set_index('id')

feature_names=[c for c in train_data.columns if 'feature' in c]

#Use PCA to reduce feature space
print('Number of features: {}'.format(len(feature_names)))
pca=PCA(n_components=NUM_COMPONENTS)
pca.fit(train_data[feature_names])

x_train_pca = pca.transform(train_data[feature_names])
x_test_pca = pca.transform(test_data[feature_names])

#First lets look at random forest
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 2000, num = 15)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_randomsearch=RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   cv=2, n_iter=100, n_jobs=-1)

rf_model=rf_randomsearch.fit(x_train_pca[:5],train_data['target'][:5])
rf_model_best=rf_model.best_estimator_

print('best model: {}'.format(model_rf.best_estimator_))
print('best score: {}'.format(model_rf.best_score_))
print('best model_params: {}'.format(model_rf.best_params_))


#Second - Gradient Boosting

loss=['ls', 'lad', 'huber', 'quantile']
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 2000, num = 15)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
learning_rate = [0.1,0.01,0.001]

random_grid = {'loss': loss,
               'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate':learning_rate}

gb = GradientBoostingRegressor()
gb_randomsearch=RandomizedSearchCV(estimator=gb, param_distributions=random_grid, n_iter=100)
gb_model=gb_randomsearch.fit(x_train_pca[:5],train_data['target'][:5])
gb_model_best=gb_model.best_estimator_

print('best model: {}'.format(model_rf.best_estimator_))
print('best score: {}'.format(model_rf.best_score_))
print('best model_params: {}'.format(model_rf.best_params_))


#Extra trees regresor

criterion=['mse', 'mae']
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 2000, num = 15)]
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'criterion': criterion,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

xtrees = ExtraTreesRegressor()
x_randomsearch=RandomizedSearchCV(estimator=xtrees,param_distributions=random_grid,n_iter=100)
x_model=x_randomsearch.fit(x_train_pca[:5],train_data['target'])

x_model_best=x_model.best_estimator_


ensemble = VotingRegressor(estimators=[('rf', rf_model_best), ('gb', gb_model_best),('x', x_model_best)])
ensemble = ensemble.fit(x_train_pca[:5], train_data['target'][:5])
predictions=ensemble.predict(test_data[feature_names])

print(predictions)








