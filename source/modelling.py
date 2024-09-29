from source.libraries import *
from source.data_transformation import Datatransformation

class DeliverySystem:
    def __init__(self, train, test):
        self.train = pd.read_csv(train)
        self.test = pd.read_csv(test)

    def data_making(self):
        transf = Datatransformation(self.train, self.test)
        self.train, self.test, self.org_train, self.org_test = transf.preprocess_the_model()

     
            



        

     
        
