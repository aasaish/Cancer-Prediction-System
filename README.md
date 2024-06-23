# Cancer-Prediction-System
The Cancer Prediction System (CPS) is an advanced machine learning model designed to predict the likelihood of cancer in patients based on a comprehensive analysis of medical data. The system leverages cutting-edge algorithms and a rich dataset of clinical records, imaging, genetic information, and lifestyle factors to provide insight on disease.
Explanation of working 
Importing Libraries:
The code starts by importing the necessary libraries: pandas for data manipulation, numpy for numerical operations, matplotlib.pyplot for data visualization, and seaborn for more advanced data visualization.
It also imports some modules from sklearn (Scikit-Learn) for machine learning tasks, such as train_test_split, LogisticRegression, and various evaluation metrics.
Loading the Data:
The code reads the 'The_Cancer_data_1500_V2.csv' file into a pandas DataFrame named df.
Time Series Analysis:
The code converts the 'Diagnosis' column from a string to an integer data type.
It then creates a 'Date' column and sets it as the index of the DataFrame, using the fixed date '2024-06-21'.
Exploratory Data Analysis (EDA):
The code prints the first few rows of the DataFrame using df.head().
It also prints information about the DataFrame using df.info() and the descriptive statistics using df.describe().
Handling Missing Values:
The code checks for any missing values in the DataFrame using df.isnull().sum().
It then drops any rows with missing values using df.dropna().
Feature Engineering:
The code creates new columns based on the existing features:
'BMI_Category': Categorizes the BMI values into 'Underweight', 'Normal', 'Overweight', and 'Obese'.
'Smoking_Status': Categorizes the 'Smoking' feature into 'Smoker' and 'Non-smoker'.
'Physical_Activity_Level': Categorizes the 'PhysicalActivity' feature into 'Low', 'Moderate', 'High', and 'Very High'.
'Alcohol_Intake_Level': Categorizes the 'AlcoholIntake' feature into 'Low', 'Moderate', 'High', and 'Very High'.


Feature Selection:
The code selects the relevant features (X) and the target variable (y) for the machine learning model.
The features include 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', and 'CancerHistory'.
The target variable is 'Diagnosis'.
Train-Test Split:
The code splits the data into training and testing sets using train_test_split() from Scikit-Learn.
The test size is set to 20% of the total data, and a random state is set for reproducibility.
Model Training:
The code creates a Logistic Regression model from Scikit-Learn and trains it on the training data using model.fit(X_train, y_train).
Model Evaluation:
The code makes predictions on the test data using model.predict(X_test) and calculates the accuracy score using accuracy_score().
It also generates a confusion matrix using confusion_matrix() and displays it using Seaborn.
Finally, it prints the classification report, which provides more detailed performance metrics such as precision, recall, F1-score, and support.
Feature Importance:
The code extracts the feature importance values from the Logistic Regression model and creates a bar plot to visualize the importance of each feature.
Taking input from user and making prediction

The code starts by prompting the user to input several pieces of information, including their age (range: 21 to 80), gender (0 for female, 1 for male), BMI (range: 16.1 to 38.8), smoking status (0 for no, 1 for yes), genetic risk level (0, 1, or 2), physical activity level (range: 1.1 to 9.5), alcohol intake level (range: 0.1 to 4.7), and history of cancer (0 for no, 1 for yes).
After collecting all the user input, the code creates a new list called new_patient that contains all the features, except for the cancer_history feature. This is because the model was likely trained on a dataset that did not include the cancer_history feature, and the code needs to match the number of features the model was trained on.
The model.predict([new_patient])) line uses the trained machine learning model to make a prediction based on the user's input. The prediction is a single value, either 0 (no cancer) or 1 (cancer).
Finally, the code prints out the prediction, using an f-string to display either "Cancer" or "No Cancer" based on the predicted value.

