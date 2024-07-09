from Data_set import FileGetter, DataProcessor, FeatureImportance
from typing import List, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt


# Create an instance of FileGetter with the file path(s)
files: Tuple[str] = FileGetter("Fargo_daily updated.csv")

# Get the list of DataFrames from the files
data_frames: List[pd.DataFrame] = files.get_data()


def plot_metrics(model_type: str, metrics: Union[float, Tuple[float, float, float, float]]):
    if model_type == "classification":
        plt.bar(["Accuracy"], [metrics])
        plt.title(f"Metrics for {model_type}")
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
    else:
        labels = ["MAE", "MSE", "RMSE", "R2"]
        plt.bar(labels, metrics)
        plt.title(f"Metrics for {model_type}")
        plt.ylim(0, max(metrics) * 1.1)
        plt.ylabel("Metric Value")
    plt.show()
        
# Create an instance of DataProcessor with the list of DataFrames
if not data_frames:
    print("No data frames to process.")
else:
    processor = DataProcessor(data_frames)
    for data_frame in processor.get_data_preprocess():
        target_column = "ST_100"  # Example: using 'ST_100' as target
        drop_columns = ["Time(CST)", "ST_50", "ST_10", "ST_100"]

        x, y = processor.preprocess(data_frame, target_column, drop_columns)

        # For classification, convert target to categorical
        for model in ["catboost", "lgbm","xgboost", "linear" ]:
            
            model_type = model
            if model_type == "classification":
                y = pd.cut(y, bins=3, labels=False)  

            feature_importance = FeatureImportance(model_type)

            top_features, metrics = feature_importance.get_feature_importance_and_metrics(x, y)
            print(f'Top Features for {model_type}:, {top_features}')
            print()
            print()
            plot_metrics(model_type, metrics)


    





