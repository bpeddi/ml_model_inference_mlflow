import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow.pyfunc
from mlflow import MlflowClient
import os 

mlflow.set_tracking_uri("http://127.0.0.1:8080")
# Sets the current active experiment to the "Apple_Models" experiment and
# returns the Experiment metadata
diabetes_experiment = mlflow.set_experiment("diabetes_model")

# Define a run name for this iteration of training.
# If this is not set, a unique name will be auto-generated for your run.
# run_name = "diabetes_rf_test"

# Define an artifact path that the model will be saved to.
artifact_path = "rf_diabete"


# # accuracy score on the test data
# def eval_metrics(actual, pred):
#     rmse = np.sqrt(mean_squared_error(actual, pred))
#     mae = mean_absolute_error(actual, pred)
#     r2 = r2_score(actual, pred)
#     return rmse, mae, r2

# diabetes_dataset = pd.read_csv('dataset/diabetes.csv') 


# df_class_0 = diabetes_dataset[diabetes_dataset["Outcome"]==0]
# df_class_1 = diabetes_dataset[diabetes_dataset["Outcome"]==1]

# print("After undersampling", len(df_class_0), len(df_class_1))

# df_class_0 = df_class_0.sample(len(df_class_1))

# print("After undersampling" , len(df_class_0), len(df_class_1))

# diabetes_dataset = pd.concat([df_class_0, df_class_1])


# # separating the data and labels
# X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
# Y = diabetes_dataset['Outcome']


# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

# with mlflow.start_run() as run:

#     classifier = svm.SVC(kernel='linear')
#     #training the support vector Machine Classifier
#     classifier.fit(X_train, Y_train)
#     # accuracy score on the training data
#     X_train_prediction = classifier.predict(X_train)
#     training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
#     X_test_prediction = classifier.predict(X_test)
#     test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
#     predicted_qualities = classifier.predict(X_test)
#     (rmse, mae, r2) = eval_metrics(Y_test, predicted_qualities)
#     alpha =  0.5
#     l1_ratio = 0.1
#     # print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
#     # print(f"  RMSE: {rmse}")
#     # print(f"  MAE: {mae}")
#     # print(f"  R2: {r2}")
#     # print('Accuracy score of the test data : ', test_data_accuracy)

#         # Log the parameters used for the model fit

#     metrics = {"mae": mae,  "rmse": rmse, "r2": r2}
#     # Log the error metrics that were calculated during validation
#     mlflow.log_metrics(metrics)

#     # Log an instance of the trained model for later use
#     model_info = mlflow.sklearn.log_model(
#         sk_model=classifier, input_example=X_test, artifact_path=artifact_path, registered_model_name="Diabetes_model"
#     )

#     my_mlpackage_dir = "my_mlpackage_dir"
#     if os.path.exists(my_mlpackage_dir):
#         try:
#             os.rmdir(my_mlpackage_dir)
#         except OSError as e:
#             print(f"Error: {my_mlpackage_dir} : {e.strerror}")
 
#     mlflow.sklearn.save_model(classifier, my_mlpackage_dir,input_example=X_test)



# loaded_model = mlflow.pyfunc.load_model("my_mlpackage_dir")  # Load model from Local File System 
loaded_model = mlflow.pyfunc.load_model(model_uri="models:/Diabetes_model/14")  # Load model from MLflow Registry
data = {
    "Pregnancies": [6, 1, 8, 1, 0],
    "Glucose": [148, 85, 183, 89, 137],
    "BloodPressure": [72, 66, 64, 66, 40],
    "SkinThickness": [35, 29, 0, 23, 35],
    "Insulin": [0, 0, 0, 94, 168],
    "BMI": [33.6, 26.6, 23.3, 28.1, 43.1],
    "DiabetesPedigreeFunction": [0.627, 0.351, 0.672, 0.167, 2.288],
    "Age": [50, 31, 32, 21, 33]
}

# Create DataFrame
df = pd.DataFrame(data)
df


# # changing the input_data to numpy array
# input_data_as_numpy_array = np.asarray(input_data)

# # reshape the array as we are predicting for one instance
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(df)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
## Get  Latest Model 

# client = MlflowClient()
# mv = client.search_model_versions("name='Diabetes_model'")
# print(f'Registered Model = {dict(list(mv)[0])["source"]}')

# mlflow models serve -m "models:/Diabetes_model/2" --port 5002
# export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
# curl -d '{"dataframe_split": {
# "columns": ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"],
# "data": [[5,125,72,19,25,17.8,0.587,51]]}}' \
# -H 'Content-Type: application/json' -X POST localhost:5002/invocations
#mlflow models build-docker -m models:/Diabetes_model/11 -n diabetes_model:latest --enable-mlserve