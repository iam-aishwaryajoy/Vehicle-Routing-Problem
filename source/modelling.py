from source.libraries import *
from source.data_transformation import Datatransformation

class DeliverySystem:
    def __init__(self, train, test):
        self.train = pd.read_csv(train)
        self.test = pd.read_csv(test)

    def data_making(self):
        transf = Datatransformation(self.train, self.test)
        self.train, self.test, self.org_train, self.org_test = transf.preprocess_the_model()
        target_column = 'Time_taken(min)'  # Replace with your actual target column name

        # For training data
        X = self.train.drop(columns=[target_column])  # Drop the target column to get features
        y = self.train[target_column]    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)              # Get the target variable

        print('Step 2: Modelling')
        print(f"    X_train shape: {X_train.shape}")
        print(f"    y_train shape: {y_train.shape}")
        print(f"    X_test shape: {X_test.shape}")
        print(f"    y_test shape: {y_test.shape if y_test is not None else 'N/A'}")

        print('    Linear regression')
        linear_regression = LinearRegression()
        linear_regression.fit(X_train, y_train)
        y_pred_LR = linear_regression.predict(X_test)

        print('    Random Forest')
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_RF = rf.predict(X_test)


        print('Step 3: Model Results:')
        print('    Linear regression')
        print('        Mean squared Error:', mean_squared_error(y_test, y_pred_LR))
        print('        R2 Score:', r2_score(y_test, y_pred_LR))

        print('    Random Forest')
        print('        Mean squared Error:', mean_squared_error(y_test, y_pred_RF))
        print('        R2 Score:', r2_score(y_test, y_pred_RF))

        print('    Random forest is the better choice.')


     
            



        

     
        
