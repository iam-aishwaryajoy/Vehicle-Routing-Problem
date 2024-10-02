from source.libraries import *
from source.utils import Save
from source.data_transformation import Datatransformation

class DeliverySystem:
    def __init__(self, train, test):
        self.train = pd.read_csv(train)
        self.test = pd.read_csv(test)

    def hypertune_random_forest(self, X_train, y_train, X_test, y_test):
        param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt'], 
        'max_depth': [10],  # Include None for no maximum depth
        'min_samples_split': [10],  # More options
        'min_samples_leaf': [4],
        'bootstrap': [True]  # Consider bootstrapping
}

        def calculate_accuracy(y_true, y_pred, tolerance=0.9):
            # Convert predictions to binary classification based on a tolerance
            is_accurate = np.abs(y_true - y_pred) <= tolerance
            return accuracy_score(np.ones_like(is_accurate), is_accurate)
        accuracy_scorer = make_scorer(calculate_accuracy)
        mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
        r2_scorer = make_scorer(r2_score)
        scoring = {'MSE': mse_scorer, 'Accuracy': accuracy_scorer, 'R2': r2_scorer}
        

        rf = RandomForestRegressor(random_state=42)
        # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring=scoring, refit='MSE')
      
        
        try:
            grid_search.fit(X_train, y_train)
        except Exception as e:
            print(f"Error during fitting: {e}")
            return

        print('    Random Forest')
        # print("        Best parameters found: ", grid_search.best_params_)
        print("        Best MSE found on Train Data: ", -grid_search.best_score_)
        best_accuracy = grid_search.cv_results_['mean_test_Accuracy'][grid_search.best_index_]
        best_r2 = grid_search.cv_results_['mean_test_R2'][grid_search.best_index_]
        print("        Best Accuracy found on Train Data: ", best_accuracy)
        print("        Best R^2 found on Train Data: ", best_r2)

        best_rf = grid_search.best_estimator_
        Save(best_rf, type='model', filename='best_model', folder_name='model/')
        y_pred = best_rf.predict(X_test)

        # Evaluate
        print("        Mean Squared Error on Test Data: ", mean_squared_error(y_test, y_pred))
        print("        R^2 Score on Test Data: ", r2_score(y_test, y_pred))
        print("        Accuracy on Test Data: ", calculate_accuracy(y_true=y_test, y_pred=y_pred))

        print('Step 4: Final Result')
        print("    Train Data Accuracy: ",round(best_accuracy*100, 2), "%")
        print("    Test Data Accuracy: ",round(calculate_accuracy(y_true=y_test, y_pred=y_pred)*100,2), "%")

        best_param = pd.DataFrame(best_rf)
        Save(best_param, type='csv', filename='best_parameters', folder_name='model/')


    def data_making(self):
        transf = Datatransformation(self.train, self.test)
        self.train, self.test, self.org_train, self.org_test = transf.preprocess_the_model()
       

    def training_model(self, alg='random_forest'):
        target_column = 'Time_taken(min)'  # Replace with your actual target column name
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.train)
        self.train = pd.DataFrame(scaled_data, columns=self.train.columns)
        X = self.train.drop(columns=[target_column])  # Drop the target column to get features
        y = self.train[target_column]    
   
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)              # Get the target variable
        
        print('Step 2: Splitting Data')
        print(f"    X_train shape: {X_train.shape}")
        print(f"    y_train shape: {y_train.shape}")
        print(f"    X_test shape: {X_test.shape}")
        print(f"    y_test shape: {y_test.shape if y_test is not None else 'N/A'}")

        print('Step 3: Model Results:')
        if alg in ['all', 'linear_regression']:
            print('    Linear regression')
            linear_regression = LinearRegression()
            linear_regression.fit(X_train, y_train)
            y_pred_LR = linear_regression.predict(X_test)
            print('    Linear regression')
            print('        Mean squared Error:', mean_squared_error(y_test, y_pred_LR))
            print('        R2 Score:', r2_score(y_test, y_pred_LR))

        if alg in ['all', 'random_forest_sample']:
            print('    Random Forest')
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_RF = rf.predict(X_test)
            print('    Random Forest')
            print('        Mean squared Error:', mean_squared_error(y_test, y_pred_RF))
            print('        R2 Score:', r2_score(y_test, y_pred_RF))
            print('    Random forest is the better choice.')

        if alg in ['all', 'random_forest']:
            self.hypertune_random_forest(X_train, y_train, X_test, y_test)
            
    def predict(self, data):
        data = pd.read_csv(data)
        loaded_model = joblib.load('model/best_model.pkl')

        transf = Datatransformation(None, data)
        test_data, org_tdata = transf.preprocess_the_model()

        y_pred = loaded_model.predict(test_data)
        test_data['Prediction'] = y_pred
        org_tdata['Prediction'] = y_pred
        Save(test_data, 'csv', filename='validation', folder_name='output/')
        Save(org_tdata, 'csv', filename='validation_org', folder_name='output/')
        print('Step 5: Prediction Completed')


     
            



        

     
        
