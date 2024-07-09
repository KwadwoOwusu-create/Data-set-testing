import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score


class FileGetter:
    def __init__(self, *file_path:  Tuple[str]) -> None:
        self.file_paths = file_path
    
    def get_data(self) -> List[pd.DataFrame]:
        data_frames = []
        for file_path in self.file_paths:
            try:
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                    data = pd.read_excel(file_path)
                elif file_path.endswith('.json'):
                    data = pd.read_json(file_path)
                elif file_path.endswith('.parquet'):
                    data = pd.read_parquet(file_path)
                elif file_path.endswith('.feather'):
                    data = pd.read_feather(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_path}")
                data_frames.append(data)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                return []
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                return []
        return data_frames
       

class DataProcessor:
    def __init__(self, data_frames: List[pd.DataFrame]) -> None:
        self.data_frames = data_frames
    
    def get_data_preprocess(self) -> List[pd.DataFrame]:
        return self.data_frames
    
    def preprocess(self, data: pd.DataFrame, target_column: str, drop_columns: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        x = data.drop(columns=drop_columns)
        y = data[target_column]
        
        for column in x.select_dtypes(include=["float64", "int64"]).columns:
            x.fillna({col: x[col].mean() for col in x.select_dtypes(include=["float64", "int64"]).columns}, inplace=True)
        
        # Fill missing values and encode categorical columns
        for col in x.select_dtypes(include=["object"]).columns:
            x[col].fillna(x[col].mode()[0], inplace=True)
            le = LabelEncoder()
            x[col] = le.fit_transform(x[col])           
        
        
        return x, y


class FeatureImportance: 
    def __init__(self, model_type: str = "regression") -> None:
        if model_type == "regression":
            self.model = RandomForestRegressor()
        elif model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "catboost":
            self.model = CatBoostRegressor(verbose=0)
        elif model_type == "xgboost":
            self.model = XGBRegressor()
        elif model_type == "lgbm":
            self.model = LGBMRegressor()
        else:
            raise ValueError(f'model_type: {model_type} is not supported')
        self.model_type = model_type
    
    def get_feature_importance_and_metrics(self, x: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Union[float, Tuple[float, float, float, float]]]:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        self.model.fit(x_train, y_train)
        predictions = self.model.predict(x_test)

        if isinstance(self.model, RandomForestClassifier):
            accuracy = accuracy_score(y_test, predictions.round())
            feature_importances = self.model.feature_importances_
            feature_importance_series = pd.Series(feature_importances, index=x.columns)
            top_features = feature_importance_series.nlargest(3).index.tolist()
            return top_features, accuracy

        elif isinstance(self.model, (RandomForestRegressor, CatBoostRegressor, XGBRegressor, LGBMRegressor)):
            feature_importances = self.model.feature_importances_
            feature_importance_series = pd.Series(feature_importances, index=x.columns)
            top_features = feature_importance_series.nlargest(3).index.tolist()
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            print(f"METRICS FOR {self.model_type}: MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}") 
            return top_features, (mae, mse, rmse, r2)

        elif isinstance(self.model, LinearRegression):
            top_features = x.columns.tolist()
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            print(f"METRICS FOR {self.model_type}: MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}") 

            return top_features[:3], (mae, mse, rmse, r2)




