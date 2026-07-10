import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():
    try:
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_path, test_path
        )

        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

        print(f"Model training completed. R2 score on test set: {r2_score:.4f}")
    except Exception as e:
        print(f"Model training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
