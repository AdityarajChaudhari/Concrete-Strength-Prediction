import os
import numpy as np
from Logging.logfile import LogClass
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from ModelTraining.model import Model
from TrainTestSplit.trainsplit import DataSplit


class ModelTuning:

    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'Train_Test_Split_log.txt'
        self.loader = DataSplit()
        self.model = Model()
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = LogClass(self.folder, self.filename)

    def txgbr(self):
        x_train, x_test, y_train, y_test = self.loader.splitdata()
        m = self.model.xgbmodel()
        params = {
            "learning_rate": [float(i) for i in np.linspace(0.001,0.3,100)],
            "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
            "min_child_weight": [1, 3, 5, 7, 8, 9],
            "gamma": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "colsample_bytree": [0.1, 0.2, 0.3, 0.4, 0.5],
            "n_estimators": [i for i in range(100,3000,100)],
            "subsample": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
        rscv_xgbr = RandomizedSearchCV(m, param_distributions = params, n_iter=50, cv = 5, n_jobs=-1, random_state=75,verbose=2)
        rscv_xgbr.fit(x_train, y_train)
        xgbr_best = rscv_xgbr.best_params_
        final_reg = XGBRegressor(n_estimators=xgbr_best['n_estimators'], learning_rate=xgbr_best['learning_rate'],
                                 min_child_weight=xgbr_best['min_child_weight'],max_depth=xgbr_best['max_depth'],
                                 gamma=xgbr_best['gamma'], colsample_bytree=xgbr_best['colsample_bytree'],subsample=xgbr_best['subsample']
                                 )
        final_reg.fit(x_train, y_train)
        print("XGB Regressor")
        print("Training Score :- ", final_reg.score(x_train, y_train))
        print("Testing Score :- ", final_reg.score(x_test, y_test))
        return final_reg

    def tgbr(self):
        x_train, x_test, y_train, y_test = self.loader.splitdata()
        gbrm = self.model.gbrmodel()
        params = {
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1],
            #'max_depth': [int(i) for i in range(1,20,1)],
            'max_features': ['auto', 'sqrt', None],
            'min_samples_leaf': [i for i in range(1, 50, 1)],
            'min_samples_split': [i for i in range(2, 50, 1)],
            'n_estimators': [int(i) for i in np.linspace(100, 3000, 50)],
            'alpha': [a for a in np.linspace(0.1, 0.99, 20)],
            'loss': ['ls', 'lad', 'huber', 'quantile']
        }
        rscv_gbr = RandomizedSearchCV(gbrm, param_distributions = params, n_iter=50, cv = 2, n_jobs=-1,verbose=True)
        rscv_gbr.fit(x_train, y_train)
        gbr_best = rscv_gbr.best_params_
        final_gbreg = GradientBoostingRegressor(learning_rate=gbr_best['learning_rate'],max_features=gbr_best['max_features'],
                                                min_samples_leaf=gbr_best['min_samples_leaf'],
                                                min_samples_split=gbr_best['min_samples_split'], n_estimators=gbr_best['n_estimators'],
                                                alpha=gbr_best['alpha'], loss=gbr_best['loss'])
        final_gbreg.fit(x_train, y_train)
        print("GB Regressor")
        print("Training Score :- ", final_gbreg.score(x_train, y_train))
        print("Testing Score :- ", final_gbreg.score(x_test, y_test))
        return final_gbreg

    def trfr(self):
        x_train, x_test, y_train, y_test = self.loader.splitdata()
        rfrm = self.model.rfrmodel()
        params = {
            'criterion': ['mse', 'friedman_mse'],
            'max_features': ['sqrt', 'auto', 'log2', None],
            'max_depth': [i for i in range(1, 50, 1)],
            'n_estimators': [i for i in range(100, 3000, 30)],
            'min_samples_split': [i for i in range(2, 50, 1)],
            'min_samples_leaf': [i for i in range(1, 50, 1)]
        }
        rscv_rfr = RandomizedSearchCV(rfrm, param_distributions=params, n_iter=50, cv=5, n_jobs=-1, verbose=1)
        rscv_rfr.fit(x_train, y_train)
        rfr_best = rscv_rfr.best_params_
        final_rfr = RandomForestRegressor(criterion=rfr_best['criterion'],max_features=rfr_best['max_features'],
                                          min_samples_leaf=rfr_best['min_samples_leaf'],max_depth=rfr_best['max_depth'],
                                          min_samples_split=rfr_best['min_samples_split'],
                                          n_estimators=rfr_best['n_estimators'],
                                        )
        final_rfr.fit(x_train, y_train)
        print("RF Regressor")
        print("Training Score :- ", final_rfr.score(x_train, y_train))
        print("Testing Score :- ", final_rfr.score(x_test, y_test))
        return final_rfr

    def tbgr(self):
        x_train, x_test, y_train, y_test = self.loader.splitdata()
        bgr = self.model.bgrmodel()
        params = {
            'n_estimators' : [i for i in range(10,300,10)],
            'max_samples' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        }
        rscv_bgr = RandomizedSearchCV(bgr, param_distributions=params, n_iter=75, cv=10, n_jobs=-1, verbose=1)
        rscv_bgr.fit(x_train, y_train)
        bgr_best = rscv_bgr.best_params_
        final_bgr = BaggingRegressor(n_estimators=bgr_best['n_estimators'],max_samples=bgr_best['max_samples'])
        final_bgr.fit(x_train, y_train)
        print("Bagging Regressor")
        print("Training Score :- ", final_bgr.score(x_train, y_train))
        print("Testing Score :- ", final_bgr.score(x_test, y_test))
        return final_bgr

    def stacking(self):
        x_train, x_test, y_train, y_test = self.loader.splitdata()
        xgbr = self.txgbr()
        gbr = self.tgbr()
        rfr = self.trfr()
        #bgr = self.tbgr()
        str = StackingRegressor([('xgbr', xgbr), ('rfr',rfr)], final_estimator=gbr)
        str.fit(x_train, y_train)
        print("stacking")
        print("Training Score :- ", str.score(x_train, y_train))
        print("Testing Score :- ", str.score(x_test, y_test))
        return str





