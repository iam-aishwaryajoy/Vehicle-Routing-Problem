from source.libraries import *
from source.preprocess import Preprocessing

class Datatransformation:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def original_data_making(self, mappings, data):
         #Train data Original making
        reordered_int_map = []
        for col, maps in mappings:
            reordered_map = {}
            for category, idx in maps.items():
                if isinstance(category, str):
                    reordered_map[idx] = category.strip()  # Strip only if it's a string
                else:
                    reordered_map[idx] = category  # Keep np.nan or other non-string values as is
            reordered_int_map.append((col, reordered_map))
        

        for tup in reordered_int_map:
            col, maps = tup
            data[col] = data[col].replace(maps)
        return data
     
        
    def preprocess_the_model(self):
        print('Step 1: Data Cleaning and Preprocessing')
        preprocess = Preprocessing(self.train, self.test)
        preprocess.drop('ID')
        preprocess.handle_nan()
        preprocess.handle_string()
        
        #Integer Mapping
        cat_cols = ['Delivery_person_ID', 'Type_of_order', 'Type_of_vehicle','Festival', 'City']
        train_mappings, test_mappings = preprocess.convert_integer_encodings(cat_cols)

        #Ordinal Encoding
        mapping = {'Sunny':1.0, 'Stormy':5.0, 'Sandstorms':6.0,'Cloudy':2.0, 'Fog':3.0, 'Windy':4.0, 'NaN':np.nan}
        original_mapping_1 = {v:k.strip() for k,v in mapping.items()}
        preprocess.convert_ordinal_encodings('Weatherconditions', mapping)
        

        mapping = {'Jam ': 3, 'High ':2, 'Medium ':1, 'Low ':0, 'NaN':np.nan}
        original_mapping_2 = {v:k.strip() for k,v in mapping.items()}
        preprocess.convert_ordinal_encodings('Road_traffic_density', mapping)
        

        preprocess.handling_missing()
        self.train, self.test = preprocess.return_data()

        #Original Data making
        print('    Original Data Is stored')
        original_train_data = self.train.copy()
        original_test_data = self.test.copy()
        
        #For ordinal Encoding Attributes
        original_train_data['Weatherconditions'] = original_train_data['Weatherconditions'].replace(original_mapping_1)
        original_test_data['Weatherconditions'] = original_test_data['Weatherconditions'].replace(original_mapping_1)
        original_train_data['Road_traffic_density'] = original_train_data['Road_traffic_density'].replace(original_mapping_2)
        original_test_data['Road_traffic_density'] = original_test_data['Road_traffic_density'].replace(original_mapping_2)
        #For Integer Encoding Attributes
        original_train_data = self.original_data_making(train_mappings, original_train_data)
        original_test_data = self.original_data_making(test_mappings, original_test_data)

       

        #Check Missing value in Original data:
        missing_train = original_train_data.isnull().sum()
        missing_test = original_test_data.isnull().sum()
        if missing_train.any():
            print("There are missing values in the original training dataset:")
            print(missing_train[missing_train > 0])   
        if missing_test.any():
            print("There are missing values in the original testing dataset:")
            print(missing_test[missing_test > 0])

        self.train.to_csv('data/final/model_train.csv', index=False )
        self.test.to_csv('data/final/model_test.csv', index=False )
        original_train_data.to_csv('data/final/original_train.csv', index=False )
        original_test_data.to_csv('data/final/original_test.csv', index=False )

        self.org_train = original_train_data
        self.org_test = original_test_data
        self.map = train_mappings

        return self.train, self.test, self.org_train, self.org_test
     
def preprocess_test_data(self):
        print('    Data Cleaning and Preprocessing')
        preprocess = Preprocessing(self.train, self.test, type='test')
        preprocess.drop('ID')
        preprocess.handle_nan()
        preprocess.handle_string()
        
        #Ordinal Encoding
        mapping = {'Sunny':1.0, 'Stormy':5.0, 'Sandstorms':6.0,'Cloudy':2.0, 'Fog':3.0, 'Windy':4.0, 'NaN':np.nan}
        original_mapping_1 = {v:k.strip() for k,v in mapping.items()}
        preprocess.convert_ordinal_encodings('Weatherconditions', mapping)
        

        mapping = {'Jam ': 3, 'High ':2, 'Medium ':1, 'Low ':0, 'NaN':np.nan}
        original_mapping_2 = {v:k.strip() for k,v in mapping.items()}
        preprocess.convert_ordinal_encodings('Road_traffic_density', mapping)
        

        # preprocess.handling_missing()
        # self.train, self.test = preprocess.return_data()

        #Original Data making
        print('    Original Data Is stored')
        original_test_data = self.test.copy()
        
        #For ordinal Encoding Attributes
        original_test_data['Weatherconditions'] = original_test_data['Weatherconditions'].replace(original_mapping_1)
        original_test_data['Road_traffic_density'] = original_test_data['Road_traffic_density'].replace(original_mapping_2)
        #For Integer Encoding Attributes
        original_test_data = self.original_data_making(self.map, original_test_data)

       

        #Check Missing value in Original data:
        missing_test = original_test_data.isnull().sum()
        if missing_test.any():
            print("There are missing values in the original testing dataset:")
            print(missing_test[missing_test > 0])

        self.test.to_csv('data/final/valid_test.csv', index=False )
        original_test_data.to_csv('data/final/original_test.csv', index=False )
        self.org_test = original_test_data

        return self.test, self.org_test