import logging 
from matplotlib import pyplot as plt 
import pandas as pd 
import os
inport mlflow
import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace  import Workspace
from azureml.core.dataset import Dataset 
from azureml.core.datastore import Datastore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score #evaluation metric
from termcolor import colored as cl
from termcolor import colored as cl

def main():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=float, help="path to input data")
    parser.add_argument("--train_data", type=float, help="path to train data")
    parser.add_argument("--test_data", type=float, help="path to test data")
    args = parser.parse_args()


    mlflow.start_run()

    import pandas as pd 
    dataset = pd.read_csv(args.data)
    dataset.drop('Time', axis = 1, inplace = True) #drop the time col since it has no effect on the training 
    print(dataset.head(3))  
    


    sc = StandardScaler() #normilze the data and print the first 10 rows
    amount = dataset['Amount'].values

    dataset['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    credit_df = pd.read_excel(args.data, header=1, index_col=0)

    mlflow.log_metric("num_samples", credit_df.shape[0])
    mlflow.log_metric("num_features", credit_df.shape[1] - 1)

    credit_train_df, credit_test_df = train_test_split(
        credit_df,
        test_size=0.2, random_state=66
    )

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    credit_train_df.to_csv(os.path.join(args.train_data, "data.csv"), index=False)

    credit_test_df.to_csv(os.path.join(args.test_data, "data.csv"), index=False)

    # Stop Logging
    mlflow.end_run()



    




if __name__ == "__main__":
    main()