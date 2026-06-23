import pandas as pd
from pathlib import Path
from io import StringIO


class DataLoader:

    def __init__(self, data_dir="data"):

        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}"
            )

    # =====================
    # CLEAN COLUMN NAMES
    # =====================

    def _fix_dataframe(self, df):

        # Хэрэв бүх өгөгдөл нэг баганад байвал
        if len(df.columns) == 1:

            col = df.columns[0]

            text = "\n".join(
                [col]
                + df.iloc[:, 0]
                .astype(str)
                .tolist()
            )

            df = pd.read_csv(
                StringIO(text),
                sep="\t"
            )

        df.columns = (
            df.columns
            .str.strip()
            .str.replace("<", "", regex=False)
            .str.replace(">", "", regex=False)
            .str.upper()
        )

        required = [
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE"
        ]

        missing = [
            c
            for c in required
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        return df

    # =====================
    # GENERIC PARQUET LOADER
    # =====================

    def _load_parquet(
        self,
        filename
    ):

        file = (
            self.data_dir /
            filename
        )

        if not file.exists():
            raise FileNotFoundError(
                f"{file} not found."
            )

        print(
            f"📂 Loading {filename}..."
        )

        df = pd.read_parquet(file)

        df = self._fix_dataframe(df)

        print(
            f"✅ Loaded {len(df):,} rows"
        )

        return df

    # =====================
    # GOLD
    # =====================

    def load_gold_data(self):

        return self._load_parquet(
            "MASTER_GOLD_10Y.parquet"
        )

    # =====================
    # EURO
    # =====================

    def load_euro_data(self):

        return self._load_parquet(
            "MASTER_EURO_10Y.parquet"
        )