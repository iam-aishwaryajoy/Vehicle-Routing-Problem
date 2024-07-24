from source.libraries import *
from source.preprocess import Preprocessing

class DeliverySystem:
    def __init__(self, train, test):
        self.train = pd.read_csv(train)
        self.test = pd.read_csv(test)
        
    def train_the_model(self):
        print('Step 1: Data Cleaning and Preprocessing')
        preprocess = Preprocessing(self.train, self.test)
        preprocess.drop('ID')
        preprocess.handle_nan()
        preprocess.handle_string()
        
        
        cat_cols = ['Delivery_person_ID', 'Type_of_order', 'Type_of_vehicle','Festival', 'City']
        preprocess.convert_integer_encodings(cat_cols)
        mapping = {'Sunny':1.0, 'Stormy':5.0, 'Sandstorms':6.0,'Cloudy':2.0, 'Fog':3.0, 'Windy':4.0, 'NaN':np.nan}
        preprocess.convert_ordinal_encodings('Weatherconditions', mapping)
        mapping = {'Jam ': 3, 'High ':2, 'Medium ':1, 'Low ':0, 'NaN':np.nan}
        preprocess.convert_ordinal_encodings('Road_traffic_density', mapping)
        preprocess.handling_missing()
        self.train, self.test = preprocess.return_data()