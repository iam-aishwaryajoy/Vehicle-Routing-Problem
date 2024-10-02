# Vehicle Routing Problem


## Project Overview
This project aims to solve the Vehicle Routing Problem (VRP) by addressing missing data using linear regression and predicting the time required for particular orders using Random Forest Algorithm. Additionally, it predicts the optimal set of routes for a fleet of vehicles to traverse to deliver to a given set of customers efficiently.

## Key Features
### Handling Missing Data:

* Utilizes linear regression to handle missing data points in the dataset.
* The linear relationship is based on correlation matrix.
* Ensures data integrity and completeness before applying predictive models.
* The data which enters into the model will not have any missing values.

### Delivery Time Prediction:

* Predicts the time required for specific orders based on historical data and relevant features.
* Helps plan and manage delivery schedules effectively.
* The Random Forest model has a significantly lower MSE (20.48) compared to Linear Regression (42.72). This suggests that the Random Forest model's predictions are closer to the actual values. The Random Forest model has an R² Score of 0.77, which is substantially better than Linear Regression's 0.51. 
* Hypertuning RF model using the mean squred error. A custom accuracy metric which is calculated based on minimum tolerance of 0.9 minutes. It indicate whether predictions falls within the difference of one minute.
* Training accuracy is improved to 91.47 and testing accuracy to 91.3%.

### Optimal Route Prediction:
* Uses advanced algorithms to determine the optimal set of routes for a fleet of vehicles.
* Minimizes travel time and distance, thereby improving delivery efficiency and reducing operational costs.
* Open Street map is used to plot map and find shortest distance.


## Project Structure
* data/: Directory containing the dataset.
* source/: Directory containing all the scripts for data preprocessing, time prediction, and route optimization.
* notebooks/: Jupyter notebooks for exploratory data analysis and model experimentation.
* models/: Directory to save and load trained models.
* requirements.txt: List of required Python libraries.
* main.py: Main File

## Model Result
* Best MSE found on Train Data:  0.2819854057829125
* Best Accuracy found on Train Data:  0.9146517187739912
* Best R^2 found on Train Data:  0.7183555591913047
* Mean Squared Error on Test Data:  0.2783250201546474
* R^2 Score on Test Data:  0.7201480397993687
* Accuracy on Test Data:  0.9135870161201887

## Getting Started
### Prerequisites
* Python 3.11 or higher
### Run
  python main.py

## Contact
For any inquiries or suggestions, please contact aishwarya.joy@outlook.in
