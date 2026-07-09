import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.expection import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.project_root = Path(__file__).resolve().parents[1]

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            data_path = self.project_root / "pipeline" / "notebook" / "StudentsPerformance.csv"
            df = pd.read_csv(data_path)
            logging.info("Read the dataset as dataframe")

            train_data_path = self.project_root / self.ingestion_config.train_data_path
            test_data_path = self.project_root / self.ingestion_config.test_data_path
            raw_data_path = self.project_root / self.ingestion_config.raw_data_path

            train_data_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(raw_data_path, index=False, header=True)

            logging.info("train test split initiatted")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(train_data_path, index=False, header=True)
            test_set.to_csv(test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return str(train_data_path), str(test_data_path)
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    