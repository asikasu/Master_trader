import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from tournament_bot.main import TournamentBot
bot = TournamentBot()
bot.run_train()
print("DONE")
