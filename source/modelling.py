from source.libraries import *
from source.data_transformation import Datatransformation

class DeliverySystem:
    def __init__(self, train, test):
        self.train = pd.read_csv(train)
        self.test = pd.read_csv(test)

    def hypertune_random_forest(self, X_train, y_train, X_test, y_test):
        # param_grid = {
        # 'n_estimators': [50, 100],  # Number of trees in the forest
        # 'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at every split
        # 'max_depth': [None, 10],  # Maximum depth of the tree
        # 'min_samples_split': [5, 10],  # Minimum number of samples required to split an internal node
        # 'min_samples_leaf': [1, 4],  # Minimum number of samples required to be at a leaf node
        # 'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
        # }
        param_grid = {
        'n_estimators': [50],  
        'max_features': ['sqrt'],  
        'max_depth': [10],  
        'min_samples_split': [10],  
        'min_samples_leaf': [1],  
        'bootstrap': [True] 
        }
        def calculate_accuracy(y_true, y_pred, tolerance=5):
            # Convert predictions to binary classification based on a tolerance
            is_accurate = np.abs(y_true - y_pred) <= tolerance
            return accuracy_score(np.ones_like(is_accurate), is_accurate)
        new_scorer = make_scorer(calculate_accuracy)
        

        rf = RandomForestRegressor(random_state=42)
        # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring=new_scorer)
      
        
        try:
            grid_search.fit(X_train, y_train)
        except Exception as e:
            print(f"Error during fitting: {e}")
            return

        print('    Random Forest')
        # print("        Best parameters found: ", grid_search.best_params_)
        print("        Best MSE found on Train Data: ", -grid_search.best_score_)

        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)

        # Evaluate
        print("        Mean Squared Error on Test Data: ", mean_squared_error(y_test, y_pred))
        print("        R^2 Score on Test Data: ", r2_score(y_test, y_pred))
        print("        Accuracy on Test Data: ", calculate_accuracy(y_true=y_test, y_pred=y_pred))


    def data_making(self, alg='random_forest'):
        transf = Datatransformation(self.train, self.test)
        self.train, self.test, self.org_train, self.org_test = transf.preprocess_the_model()
        target_column = 'Time_taken(min)'  # Replace with your actual target column name

        # For training data
        X = self.train.drop(columns=[target_column])  # Drop the target column to get features
        y = self.train[target_column]    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)              # Get the target variable

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

        # if alg in ['all', 'random_forest']:
        #     print('    Random Forest')
        #     rf = RandomForestRegressor(n_estimators=100, random_state=42)
        #     rf.fit(X_train, y_train)
        #     y_pred_RF = rf.predict(X_test)
            # print('    Random Forest')
            # print('        Mean squared Error:', mean_squared_error(y_test, y_pred_RF))
            # print('        R2 Score:', r2_score(y_test, y_pred_RF))
            # print('    Random forest is the better choice.')

        if alg in ['all', 'random_forest']:
            self.hypertune_random_forest(X_train, y_train, X_test, y_test)


     
            



        

     
        
