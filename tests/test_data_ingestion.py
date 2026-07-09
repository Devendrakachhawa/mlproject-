import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.components.data_ingestion import DataIngestion


def test_data_ingestion_creates_artifacts(tmp_path):
    os.chdir(tmp_path)
    try:
        train_path, test_path = DataIngestion().initiate_data_ingestion()
        assert os.path.exists(train_path)
        assert os.path.exists(test_path)
        assert os.path.exists(os.path.join(tmp_path, "artifacts", "data.csv"))
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        assert len(train_df) + len(test_df) > 0
    finally:
        os.chdir(Path(__file__).resolve().parents[1])
        shutil.rmtree(tmp_path, ignore_errors=True)
