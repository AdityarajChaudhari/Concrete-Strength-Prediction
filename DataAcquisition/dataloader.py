import pandas as pd
from Logging.logfile import LogClass
import os


class DataAccess:

    """

    ClassName  : DataAccess
    Description: This class is used to acquire/access data that is stored in the csv format in one folder.
    Written by : Adityaraj Hemant Chaudhari
    Version    : 0.1
    Revisions  : 0

    """

    def __init__(self):
        self.data_src = r'../Data/concrete_data.csv'
        self.folder = '../LogFiles/'
        self.filename = 'DataAcquisition_log.txt'

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = LogClass(self.folder, self.filename)

    def get_data(self):

        """

        Method_Name : get_data
        Description : This method is used to acquire the data from the data source
        Output      : Pandas DataFrame
        On_Failure  : Raise Exceptions

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : 0

        """

        try:
            self.log_object.create_log_file("""Loading training data set from the local source into pandas DataFrame""")
            data = pd.read_csv(self.data_src)
            return data
        except Exception as e:
            self.log_object.create_log_file("The error is :- " + str(e))
            raise e




