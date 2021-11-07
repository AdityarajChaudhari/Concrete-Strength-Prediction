import pandas as pd
import os
import pickle
from Logging.logfile import LogClass
from sklearn.preprocessing import StandardScaler
from DataSplitting.Splitter import Splitter


class Scaling:

    """

    Class_Name : Scaler
    Description: This Class is used to scaling the dependent features.
    Written By : Adityaraj Hemant Chaudhari
    Version    : 0.1
    Revisions  : None

    """

    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'Scaler_log.txt'
        self.loader = Splitter()
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = LogClass(self.folder, self.filename)

    def scaledata(self):

        """

        Method_Name : scaledata
        Description : This method is used to scale the dependent features.
        output      : DataFrame
        on failure  : raise exception

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.log_object.create_log_file("Accessing the Dependent & Independent Data")
            x, y = self.loader.split()
            self.log_object.create_log_file("Using StandardScaler to Scale down the data")
            scalar = StandardScaler()
            x = pd.DataFrame(scalar.fit_transform(x), columns=x.columns)
            self.log_object.create_log_file("Data Scaled Successfully!!")
            return x, y, scalar
        except Exception as e:
            self.log_object.create_log_file("The Error is :- ", e)
            raise e

    def serializescaler(self):

        """

        Method_Name : serializescaler
        Description : This function is used to save Robust Scaler .
        output      : Pickle format
        on failure  : raise exception

        Written by  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.log_object.create_log_file("Saving the Scaler in the Serialized manner i.e in Pickle Format")
            a, b, std = self.scaledata()
            pickle.dump(std, open('Scaler.pkl', 'wb'))
            self.log_object.create_log_file('StandardScaler Saved in pickle format')
        except Exception as e:
            self.log_object.create_log_file("The Error is :- ", e)
            raise e











