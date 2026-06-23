class RiskManager:

    def get_lot_size(self, probability):

        if probability >= 0.95:
            return 0.10

        if probability >= 0.90:
            return 0.05

        if probability >= 0.85:
            return 0.03

        if probability >= 0.80:
            return 0.02

        return 0.01