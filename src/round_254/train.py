import sys
sys.path.append("..") 
import argparse
import pickle
import json

import numerapi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor

import utilities
import evaluation
from autoencoder import AutoEncoder

PARAMS={'num_leaves': [30, 40, 50], 
        'max_depth': [4, 5, 6], 
        'learning_rate': [0.05, 0.01, 0.005],
        'bagging_freq':[7], 
        'bagging_fraction': [0.6, 0.7, 0.8], 
        'feature_fraction': [0.85, 0.75, 0.65]}

OPT = Adam()
LOSS = mse

def train():
    
    try:
        train_data=utilities.get_data(TRAIN_PATH)
        test_data=utilities.get_data(TEST_PATH)
    except Exception as e:
        print(e)
        num_api = numerapi.NumerAPI(PUBLIC_KEY, SECRET_GUY,verbosity="info")
        num_api.download_current_dataset(dest_path='../data/')
        feature_names=utilities.get_feature_names(TRAIN_PATH)
        train_data=utilities.get_data(TRAIN_PATH)
        test_data=utilities.get_data(TEST_PATH)

    feature_names=utilities.get_feature_names(train_data)
    x_train=train_data[feature_names]
    x_test=test_data[feature_names]
    #call autoencoder for dimensionality reduction
    ae=AutoEncoder(x_train.shape,N_COMPONENTS)
    model=ae.build()
    model.compile(optimizer=OPT, loss=LOSS)
    history=model.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, validation_data=(x_test,x_test))
    
    #get the autoencoder representation
    x_train_ae = model.predict(x_train)
    x_test_ae = model.predict(x_test)

    #corrupt dataset using gaussian noise
    #mu,sigma=0,0.1
    #noise=np.random.normal(mu,sigma,x_train_pca.shape)
    #x_train_pca_noise=x_train_pca+noise

    #train an LGBMRegressor model - use random search for parameter tuning
    #with cross validation
    lgb=LGBMRegressor()
    lgb_randomsearch=RandomizedSearchCV(estimator=lgb,cv=CV,param_distributions=params, n_iter=100)
    lgb_model=lgb_randomsearch.fit(x_train_ae[:100],train_data['target'][:100])
    lgb_model_best=lgb_model.best_estimator_
    lgb_model_best=lgb_model_best.fit(x_train_ae[:100],train_data['target'][:100])
    
    print("Generating all predictions...")
    train_data['prediction'] = lgb_model.predict(x_train_ae)
    test_data['prediction'] = lgb_model.predict(x_test_ae)

    train_corrs = (evaluation.per_era_score(train_data))
    print('train correlations mean: {}, std: {}'.format(train_corrs.mean(), train_corrs.std(ddof=0)))
    #print('avg per-era payout: {}'.format(evaluation.payout(train_corrs).mean()))

    valid_data = test_data[test_data.data_type == 'validation']
    valid_corrs = evaluation.per_era_score(valid_data)
    #valid_sharpe = evaluation.sharpe(valid_data)
    print('valid correlations mean: {}, std: {}'.format(valid_corrs.mean(), valid_corrs.std(ddof=0)))
    #print('avg per-era payout {}'.format(evaluation.payout(valid_corrs.mean())))
    #print('valid sharpe: {}'.format(valid_sharpe))

    #live_data = test_data[test_data.data_type == "test"]
    #live_corrs = evaluation.per_era_score(test_data)
    #test_sharpe = evaluation.sharpe(test_data)
    #print('live correlations - mean: {}, std: {}'.format(live_corrs.mean(),live_corrs.std(ddof=0)))
    #print('avg per-era payout is {}'.format(evaluation.payout(live_corrs).mean()))
    #print('live Sharpe: {}'.format(test_sharpe))
    
    #pickle and save the model
    with open('lgbm_model_round_253.pkl', 'wb') as f:
        pickle.dump(lgb_model,f)

    #save down predictions
    valid_corrs.to_csv('valid_predictions.csv')


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('-cp','--configpath',required=True,help='config path')

    args = vars(ap.parse_args())
    config_path=args['configpath']
    with open(config_path) as json_file:
        config=json.load(json_file)
    
    PUBLIC_KEY=config['public_key']
    SECRET_KEY=config['secret_key']
    MODEL_ID=config['model_id']

    TRAIN_PATH=config['train_path']
    TEST_PATH=config['test_path']
   
    #model parameters 
    N_COMPONENTS=config['n_components']
    CV=config['cv_folds']
    BATCH_SIZE=config['batch_size']
    EPOCHS=config['epochs']

    train()
