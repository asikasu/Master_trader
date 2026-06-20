# core/data_loader.py
import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_parquet(self, filename: str) -> pd.DataFrame:
        """Parquet форматаар хадгалсан датаг унших"""
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"{filename} олдсонгүй: {file_path}")
        return pd.read_parquet(file_path)

# Тест хийх хэсэг
if __name__ == "__main__":
    loader = DataLoader("data")
    print("DataLoader бэлэн боллоо.")