from source.libraries import *

class Preprocessing:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.train_corr = {}
        self.test_corr = {}
    
    def drop(self, col):
        self.train.drop(col, axis=1, inplace=True)
        self.test.drop(col, axis=1, inplace=True)
        print(f'    Dropped {col}')
    
    def handle_string(self):
        self.train['Time_taken(min)'] = self.train['Time_taken(min)'].str.replace('(min)','', regex=False) 
        self.train['Time_taken(min)'] = self.train['Time_taken(min)'].astype(int)   

        self.train['Weatherconditions'] = self.train['Weatherconditions'].str.replace('conditions ','', regex=False) 
        self.test['Weatherconditions'] = self.test['Weatherconditions'].str.replace('conditions ','', regex=False) 

        self.train['Order_Date'] = self.train['Order_Date'].str.replace('-', '', regex=True)
        self.test['Order_Date'] = self.test['Order_Date'].str.replace('-', '', regex=True)

        self.train['Time_Order_picked'] = self.train['Time_Order_picked'].str.replace(':', '', regex=True)
        self.test['Time_Order_picked'] = self.test['Time_Order_picked'].str.replace(':', '', regex=True)

        self.train['Time_Orderd'] = self.train['Time_Orderd'].str.replace(':', '', regex=True)
        self.test['Time_Orderd'] = self.test['Time_Orderd'].str.replace(':', '', regex=True)       

    
        
    def convert_integer_encodings(self, cols):
        for col in cols:
            category_to_integer = {category: idx for idx, category in enumerate(self.train[col].unique())}
            self.train[col] = self.train[col].map(category_to_integer)
        
        for col in cols:
            category_to_integer = {category: idx for idx, category in enumerate(self.test[col].unique())}
            self.test[col] = self.test[col].map(category_to_integer)
    
    def handle_nan(self):
        self.train.replace('NaN ', np.nan, inplace=True)
        self.test.replace('NaN ', np.nan, inplace=True)

        self.train.replace('condition NaN ', np.nan, inplace=True)
        self.test.replace('condition NaN ', np.nan, inplace=True)

        
        
    def convert_ordinal_encodings(self, col, size_mapping):
        self.train[col] = self.train[col].map(size_mapping)
        self.test[col] = self.test[col].map(size_mapping)
    
    def correlation(self, data, X):
        corr = data.corr()   
        corr = corr[X]
        corr = corr.drop(X, axis=0)
        corr = corr.to_frame(name='Correlation')  # Convert Series to DataFrame for easy manipulation
        corr['Absolute_Correlation'] = corr['Correlation'].abs()
        max_value = corr['Absolute_Correlation'].max()
        max_variable = corr['Absolute_Correlation'].idxmax()
        return max_variable, max_value, corr
    
    def regression_prediction(self, data, Y, X):
        '''  
        This function is used to fill missing data using regression approach. The data has X component which is the input and Y' component which is the output of regression model.
        Using X data it will predict Y'. And fill those predicted values into the missing values of Y column. 
        
        '''
        data = data[[X,Y]].copy()
        count = data[X].isna().sum()
        
        #Handling missing values in X:
        if count > 0:
            data[X] = pd.to_numeric(data[X], errors='coerce') 
            mean_value = data[X].mean() 
            data.loc[:, X] = data[X].fillna(mean_value)
            print(f'    Train Data Filling Gaps for {X} for predicting {Y}')

        train_data = data.dropna(subset=[Y])
        predict_data = data[data[Y].isnull()]  # Data that needs prediction (missing 'Y')
        predict_data = predict_data[[X]].dropna() #Dropped X that is null

        # Initialize the regression model
        model = LinearRegression()
        model.fit(train_data[[X]], train_data[Y])
        predicted_values = model.predict(predict_data)
        data.loc[data[Y].isnull(), Y] = predicted_values

        return data
    
    def convert_num_train(self, X):
        self.train[X] = self.train[X].astype(int)
    
    def convert_num_test(self, X):
        self.test[X] = self.test[X].astype(int)
        

    def handling_missing(self):
        def handle(data,  missing_data, train):
            ''' The X and Y component is selected based on correlation map'''

            self.flag = False
            self.train_map = {'Delivery_person_ID':'Delivery_location_latitude',
                            'Delivery_person_Age':'Time_taken(min)', 'Road_traffic_density':'Time_taken(min)',
                            'multiple_deliveries': 'Time_taken(min)','Festival':'Time_taken(min)', 'City':'Time_taken(min)',
                            'Delivery_person_Ratings':'Time_taken(min)','Time_taken(min)':'Road_traffic_density',
                            'Type_of_order':'multiple_deliveries','Type_of_vehicle':'Vehicle_condition',
                            'Vehicle_condition':'Type_of_vehicle', 'Weatherconditions':'Delivery_person_Ratings',
                            'Restaurant_latitude':'Delivery_location_latitude', 'Restaurant_longitude':'Delivery_location_longitude',
                            'Delivery_location_longitude':'Restaurant_longitude', 'Delivery_location_latitude':'Restaurant_latitude',
                            'Order_Date':'Time_Orderd', 'Time_Order_picked':'Time_Orderd',
                            'Time_Orderd':'Time_Order_picked','Delivery_person_ID':'Delivery_location_latitude'}
            
            self.test_map = {'Delivery_person_ID':'Delivery_location_longitude', 
                            'Road_traffic_density':'Festival','City':'Road_traffic_density',
                            'Festival':'Road_traffic_density',
                            'Type_of_vehicle':'Vehicle_condition',
                            'Restaurant_latitude':'Delivery_location_latitude', 'Restaurant_longitude':'Delivery_location_longitude',
                            'Delivery_location_longitude':'Restaurant_longitude', 'Delivery_location_latitude':'Restaurant_latitude',
                            'Vehicle_condition':'Type_of_vehicle',

                            'Order_Date':'Time_Orderd', 
                            'Time_Order_picked':'Time_Orderd',
                            'Time_Orderd':'Road_traffic_density', 
                            'Delivery_person_Age':'multiple_deliveries',
                            'multiple_deliveries': 'Road_traffic_density',
                            'Delivery_person_Ratings':'multiple_deliveries',
                            'Type_of_order':'multiple_deliveries',#'Delivery_location_longitude'
                            'Weatherconditions':'Delivery_person_Ratings', #Featival
                            }
                            #City - Multiple Deliveris

                    
            for feature, count in missing_data.items():
                if count >0:
                    self.flag = True
                    if train:           
                        independent_var = self.train_map[feature]
                    else:
                        independent_var = self.test_map[feature]

                    update = self.regression_prediction(data, feature, independent_var)
                    if feature == 'Delivery_person_Ratings':
                        data[feature] = update[feature].astype(float)
                    else:
                        data[feature] = update[feature].astype(int)                    
                            

            return data
                
        
        missing_train = self.train.isnull().sum()
        missing_test = self.test.isnull().sum()
        
        # self.convert_num()
        # self.convert_num()
        # self.convert_num()
        self.train = handle(self.train, missing_train, train=True)
        self.convert_num_train('Delivery_person_Age')
        self.convert_num_train('Order_Date')
        self.convert_num_train('Time_Order_picked')
        for feature in self.train.columns:
            attr, score, corr = self.correlation(self.train, feature) 
            self.train_corr[feature] = attr

        
        self.test = handle(self.test, missing_test, train=False)
        for feature in self.test.columns:
            attr, score, corr = self.correlation(self.test, feature) 
            self.test_corr[feature] = attr

        if self.flag == False:
                print('    Handled Missing value without alteration!')
        

    def return_data(self):
        # Check if there are any missing values in the testing set
        missing_train = self.train.isnull().sum()
        missing_test = self.test.isnull().sum()
        if missing_train.any():
            print("There are missing values in the training dataset:")
            print(missing_train[missing_train > 0])   
        if missing_test.any():
            print("There are missing values in the testing dataset:")
            print(missing_test[missing_test > 0])

        self.train.to_csv('data/interim/train.csv', index=False)
        self.train.to_csv('data/interim/test.csv', index=False)
        return self.train, self.test