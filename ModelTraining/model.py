import os
from Logging.logfile import LogClass
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from TrainTestSplit.trainsplit import DataSplit


class Model:

    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'Train_Test_Split_log.txt'
        self.loader = DataSplit()
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = LogClass(self.folder, self.filename)

    def rfrmodel(self):
        x_train, x_test, y_train, y_test = self.loader.splitdata()
        rfr = RandomForestRegressor()
        rfr.fit(x_train, y_train)
        print("Random Forest Regressor :- ")
        print("Training Score :- ", rfr.score(x_train, y_train))
        print("Testing Score :- ", rfr.score(x_test,y_test))
        return rfr

    def etrmodel(self):
        x_train, x_test, y_train, y_test = self.loader.splitdata()
        etr = ExtraTreesRegressor()
        etr.fit(x_train, y_train)
        print("ExtraTrees Regressor")
        print("Training Score :- ", etr.score(x_train, y_train))
        print("Testing Score :- ", etr.score(x_test, y_test))
        return etr

    def xgbmodel(self):
        x_train, x_test, y_train, y_test = self.loader.splitdata()
        xgbr = XGBRegressor()
        xgbr.fit(x_train, y_train)
        print("XGB Regressor")
        print("Training Score :- ", xgbr.score(x_train, y_train))
        print("Testing Score :- ", xgbr.score(x_test, y_test))
        return xgbr

    def bgrmodel(self):
        x_train, x_test, y_train, y_test = self.loader.splitdata()
        bgr = BaggingRegressor()
        bgr.fit(x_train, y_train)
        print("Bagging Regressor")
        print("Training Score :- ", bgr.score(x_train, y_train))
        print("Testing Score :- ", bgr.score(x_test, y_test))
        return bgr

    def gbrmodel(self):
        x_train, x_test, y_train, y_test = self.loader.splitdata()
        gbr = GradientBoostingRegressor()
        gbr.fit(x_train, y_train)
        print("Gradient Boosting Regressor")
        print("Training Score :- ", gbr.score(x_train, y_train))
        print("Testing Score :- ", gbr.score(x_test, y_test))
        return gbr



