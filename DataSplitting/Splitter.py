import os
from Logging.logfile import LogClass
from DataPreProcessing.FeatureEngg import FeatEng


class Splitter:

    """

    Class_Name : Splitter
    Description: This Class is used to Split the data into Dependent and Independent Features.
    Written By : Adityaraj Hemant Chaudhari
    Version    : 0.1
    Revisions  : None

    """

    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'Splitter_log.txt'
        self.loader = FeatEng().drop()
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = LogClass(self.folder, self.filename)

    def split(self):

        """

        Method_Name : split
        Description : This method is used to split Dataset into Dependent & Independent sets
        output      : DataFrame
        on failure  : raise exception

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.log_object.create_log_file("Accessing the data")
            data = self.loader
            self.log_object.create_log_file("splitting the data into Dependent and Independent Set")
            x = data.drop('concrete_compressive_strength', axis=1)
            y = data['concrete_compressive_strength']
            self.log_object.create_log_file("Data Splitted into Dependent & Independent Set")
            return x, y
        except Exception as e:
            self.log_object.create_log_file("The Error is :- ", e)
            raise e


