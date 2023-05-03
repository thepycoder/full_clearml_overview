import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import plot_tree

from clearml.automation.controller import PipelineDecorator
from clearml import Task, Dataset, TaskTypes


@PipelineDecorator.component(return_values=['X_train', 'y_train', 'X_test', 'y_test'], cache=True, task_type=TaskTypes.data_processing)
def prepare_data(dataset_name):
    # Imports first
    from clearml import Dataset
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    
    # Read the data
    data_path = Dataset.get(dataset_name=dataset_name, alias=dataset_name).get_local_copy()
    fashion_mnist_test = pd.read_csv(f"{data_path}/fashion-mnist_test.csv")
    fashion_mnist_train = pd.read_csv(f"{data_path}/fashion-mnist_train.csv")

    # Load in the train and test sets
    X_train = np.array(fashion_mnist_train.iloc[:,1:])
    y_train = np.array(fashion_mnist_train.iloc[:,0])
    X_test = np.array(fashion_mnist_test.iloc[:,1:])
    y_test = np.array(fashion_mnist_test.iloc[:,0])

    # Plot one of them to make sure everything is alright
    plt.imshow(X_train[1].reshape((28, 28)))
    plt.title("Sample Image")
    plt.show()

    return X_train, y_train, X_test, y_test


@PipelineDecorator.component(return_values=['model'], task_type=TaskTypes.training)
def train_model(X_train, y_train):
    # Imports first
    import xgboost as xgb
    from clearml import Task
    
    # Load the data into XGBoost format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    # Set the parameters
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 4,  # the maximum depth of each tree
        "eta": 0.3,  # the training step for each iteration
        "gamma": 0,
        "max_delta_step": 1,
        "subsample": 1,
        "sampling_method": "uniform",
        "seed": 42
    }
    Task.current_task().connect(params)

    # Train the XGBoost Model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=25,
        evals=[(dtrain, "train")],
        verbose_eval=0,
    )

    # Save the model
    model.save_model("best_model")
    
    return model


@PipelineDecorator.component(return_values=['accuracy'], cache=True, task_type=TaskTypes.qc)
def evaluate_model(model, X_test, y_test):
    # Imports first
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    import xgboost as xgb
    from xgboost import plot_tree
    
    
    # Load the data in XGBoost format
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Make predictions for test data
    y_pred = model.predict(dtest)
    predictions = [round(value) for value in y_pred]

    # Evaluate predictions
    accuracy = accuracy_score(dtest.get_label(), predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # Plots
    plot_tree(model)
    plt.title("Decision Tree")
    plt.show()
    
    return accuracy


@PipelineDecorator.pipeline(name='Simple Pipeline', project='Full Overview', version='0.0.5')
def run_pipeline(dataset_name):
        # Imports first
        from clearml import Task

        # Get the data in XGBoost format
        X_train, y_train, X_test, y_test = prepare_data(dataset_name=dataset_name)
        
        # Train an XGBoost model on the data
        model = train_model(X_train, y_train)
        
        # Evaluate the model
        accuracy = evaluate_model(model, X_test, y_test)
        Task.current_task().get_logger().report_single_value(name="Accuracy", value=accuracy)
        
        return accuracy



if __name__ == "__main__":
    PipelineDecorator.run_locally()
    run_pipeline(dataset_name="Fashion MNIST")