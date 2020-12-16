import math


class MCTSGame:
    def __init__(self,
                 ucb_constant: float = math.sqrt(2),
                 early_stop_frac: float = 0.1) -> None:
        self.early_stop_frac = early_stop_frac
        self.ucb_constant = ucb_constant