import pandas as pd
import numpy as np
import os
from Logging.logfile import LogClass
from DataAcquisition.dataloader import DataAccess
pd.set_option("display.max_columns",None)


class FeatEng:

    """

    Class_Name : FeatEngg
    Description: This Class is used to Perform Feature Engineering.
    Written By : Adityaraj Hemant Chaudhari
    Version    : 0.1
    Revisions  : None

    """

    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'Preprocessing_log.txt'
        self.loader = DataAccess()
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = LogClass(self.folder, self.filename)

    def replace(self):

        """

        Method_Name : replace
        Description : This method is used to replace the rows containing 0 to appropriate values
        output      : DataFrame
        on failure  : raise exception

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.log_object.create_log_file("Accessing the data")
            data = self.loader.get_data()
            self.log_object.create_log_file("Replacing some of the values")

            def replacezeros(col, to_replace, val):
                data[col] = data[col].replace(to_replace,val)

            replacezeros('blast_furnace_slag', 0, 5)
            replacezeros('fly_ash', 0, 5)
            replacezeros('superplasticizer', 0, 2)
            self.log_object.create_log_file("Replaced the zeros with appropriate values")
            return data
        except Exception as e:
            self.log_object.create_log_file("The error is : - ", e)
            raise e

    # def scale(self):
    #
    #     """
    #
    #     Method_Name : scale
    #     Description : This method is used to scale the data using logarithmic transformation
    #     output      : DataFrame
    #     on failure  : raise exception
    #
    #     Written by  : Adityaraj Hemant Chaudhari
    #     Version     : 0.1
    #     Revisions   : None
    #
    #     """
    #
    #     try:
    #         self.log_object.create_log_file("Accessing the new data")
    #         data = self.replace()
    #         self.log_object.create_log_file("Performing Logarithmic Transformation to handle Outliers")
    #
    #         def logscale(col):
    #             data[col] = np.log2(data[col])
    #         logscale('cement')
    #         logscale('blast_furnace_slag')
    #         logscale('fly_ash')
    #         logscale('superplasticizer')
    #         self.log_object.create_log_file("Log Transformation Successful!!")
    #         return data
    #     except Exception as e:
    #         self.log_object.create_log_file("The error is :- ", e)
    #         raise e

    def derivefeat(self):

        """

        Method_Name : derivefet
        Description : This method is used to transform the data into meaningful data.
        output      : DataFrame
        on failure  : raise exception

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.log_object.create_log_file("Accessing the Previously transformed data")
            data = self.replace()
            self.log_object.create_log_file("Deriving new features")

            def replace_values(col, to_replace, value):
                data[col] = data[col].replace(to_replace, value)
            self.log_object.create_log_file("some range of value to be replaced")
            replace_values('age', [1, 3], '3 or Less Than 3 Days')
            replace_values('age', 7, '1 Week')
            replace_values('age', 14, '2 Weeks')
            replace_values('age', 28, '4 Weeks')
            replace_values('age', 56, '8 Weeks')
            replace_values('age', [90, 91], '13 Weeks')
            replace_values('age', [100, 120, 180, 270, 360, 365], 'More than 13 Weeks')
            self.log_object.create_log_file("Values Replaced Successfully!!")
            return data
        except Exception as e:
            self.log_object.create_log_file("The error is :- ", e)
            raise e

    def encoder(self):

        """

        Method_Name : encoder
        Description : This method is used to transform the categorical values into numerical values using One Hot Encoding
        output      : DataFrame
        on failure  : raise exception

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.log_object.create_log_file("Access the Previously transformed data")
            data = self.derivefeat()
            self.log_object.create_log_file("Performing one hot encoding on data")
            x = pd.get_dummies(data['age'], drop_first=True)
            data = pd.concat([data,x],axis=1)
            self.log_object.create_log_file("One hot Encoding Successful!!")
            return data
        except Exception as e:
            self.log_object.create_log_file("the error is :- ", e)
            raise e

    def drop(self):

        """

        Method_Name : drop
        Description : This method is used to drop the columns
        output      : DataFrame
        on failure  : raise exception

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.log_object.create_log_file("Accessing the Previously Transformed data")
            data = self.encoder()
            self.log_object.create_log_file("Drop the Column")
            data.drop('age',axis=1,inplace=True)
            self.log_object.create_log_file("The Column Dropped Successfully!!")
            return data
        except Exception as e:
            self.log_object.create_log_file("The error is :- ",e)
            raise e
























