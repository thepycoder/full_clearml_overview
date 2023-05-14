from clearml import Dataset, Task, TaskTypes
import numpy as np
import xgboost as xgb
import pandas as pd

task = Task.init(
    project_name="Full Overview",
    task_name="Batch Inference",
    task_type=TaskTypes.inference
)

# Read the data
data_path = Dataset.get(dataset_name="Fashion MNIST", alias="Fashion MNIST").get_local_copy()
fashion_mnist_test = pd.read_csv(f"{data_path}/fashion-mnist_test.csv")
X_test = np.array(fashion_mnist_test.iloc[:,1:])
y_test = np.array(fashion_mnist_test.iloc[:,0])
dtest = xgb.DMatrix(X_test, label=y_test)

# Get the model
task_id = "2ee1ba0c51eb46c4a0b9329fc9b46a1f"
task.set_parameter("task_id", task_id)
model_path = Task.get_task(task_id=task_id).models.get('output')[0].get_local_copy()
bst = xgb.Booster()
bst.load_model(model_path)

# Make predictions for test data
y_pred = bst.predict(dtest)
predictions = [round(value) for value in y_pred]

# Save the predictions on the task itself
task.upload_artifact('Predictions', artifact_object=predictions)