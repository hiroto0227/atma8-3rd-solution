from pathlib import Path
import datetime

DATA_DIR = Path(__file__).resolve().parents[1].joinpath("atma8/data/")

JST = datetime.timezone(datetime.timedelta(hours=+9), "JST")
DATE = datetime.datetime.now()

LGB_MODEL = "lgb.model"
SUBMISSION_CSV = f"{DATE.strftime('%Y-%m-%d-%H:%M:%S')}_submission.csv"

SEED = 42
