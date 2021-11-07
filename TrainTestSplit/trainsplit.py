import os
from Logging.logfile import LogClass
from DataScaler.scaler import Scaling
from sklearn.model_selection import train_test_split


class DataSplit:

    """

    ClassName  : DataSplit
    Description: This class is used to split the dependent and independent features into training and testing set.
    Written By : Adityaraj Hemant Chaudhari.
    Version    : 0.1
    Revisions  : None

    """

    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'Train_Test_Split_log.txt'
        self.loader = Scaling()
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = LogClass(self.folder, self.filename)

    def splitdata(self):

        """

        Method_Name : split
        Description : Splitting the dependent and independent dataset into training and testing set.
        Output      : DataFrame
        On Failure  : Raise Exception

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revision    : None

        """

        try:
            #self.log_object.create_log_file("Accessing the data to Split")
            x, y, rob = self.loader.scaledata()
            #self.log_object.create_log_file("Splitting the data into training and testing set")
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.275, random_state=101)
            #self.log_object.create_log_file("Data Splitted into training and testing set successfully!!")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            self.log_object.create_log_file("The Error is :- ", e)
            raise e


d = DataSplit()
d.splitdata()