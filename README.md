# Vehicle Routing Problem


## Project Overview
This project aims to solve the Vehicle Routing Problem (VRP) by addressing missing data using linear regression and predicting the time required for particular orders using Random Forest Algorithm. Additionally, it predicts the optimal set of routes for a fleet of vehicles to traverse to deliver to a given set of customers efficiently.

## Key Features
### Handling Missing Data:

* Utilizes linear regression to handle missing data points in the dataset.
* The linear relationship is based on correlation matrix.
* Ensures data integrity and completeness before applying predictive models.

### Time Prediction:

* Predicts the time required for specific orders based on historical data and relevant features.
* Helps plan and manage delivery schedules effectively.
* The Random Forest model has a significantly lower MSE (20.48) compared to Linear Regression (42.72). This suggests that the Random Forest model's predictions are closer to the actual values. The Random Forest model has an R² Score of 0.77, which is substantially better than Linear Regression's 0.51. 

### Optimal Route Prediction:
* Uses advanced algorithms to determine the optimal set of routes for a fleet of vehicles.
* Minimizes travel time and distance, thereby improving delivery efficiency and reducing operational costs.

## Getting Started
### Prerequisites
* Python 3.11 or higher

## Project Structure
* data/: Directory containing the dataset.
* source/: Directory containing all the scripts for data preprocessing, time prediction, and route optimization.
* notebooks/: Jupyter notebooks for exploratory data analysis and model experimentation.
* models/: Directory to save and load trained models.
* requirements.txt: List of required Python libraries.
* main.py: Main File

## Run
  python main.py

## Contact
For any inquiries or suggestions, please contact aishwarya.joy@outlook.in
