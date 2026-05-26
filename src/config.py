from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = Path(r"C:\Users\LAI\Documents\Codex\influenza dataset")
RAW_FLUNET_CSV = RAW_DATA_DIR / "VIW_FNT.csv"
RAW_METADATA_CSV = RAW_DATA_DIR / "VIW_FLU_METADATA.csv"

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
APP_DIR = PROJECT_ROOT / "app"

TARGET_COUNTRIES = {
    "ARE": "United Arab Emirates",
    "MYS": "Malaysia",
}

TRANSFER_POOLS = {
    "ARE": ["OMN", "QAT", "BHR", "SAU", "JOR", "IRQ", "IRN", "PAK", "AFG", "LBN", "PSE"],
    "MYS": ["THA", "SGP", "PHL", "VNM", "KHM", "IDN", "LAO", "BRN", "MMR", "TLS"],
}

NUMERIC_COLUMNS = [
    "SPEC_PROCESSED_NB",
    "SPEC_RECEIVED_NB",
    "AH1N12009",
    "AH1",
    "AH3",
    "AH5",
    "AH7N9",
    "ANOTSUBTYPED",
    "ANOTSUBTYPABLE",
    "AOTHER_SUBTYPE",
    "INF_A",
    "BVIC_2DEL",
    "BVIC_3DEL",
    "BVIC_NODEL",
    "BVIC_DELUNK",
    "BYAM",
    "BNOTDETERMINED",
    "INF_B",
    "INF_ALL",
    "INF_NEGATIVE",
    "ILI_ACTIVITY",
    "ADENO",
    "BOCA",
    "HUMAN_CORONA",
    "METAPNEUMO",
    "PARAINFLUENZA",
    "RHINO",
    "RSV_PROCESSED",
    "RSV",
    "OTHERRESPVIRUS",
]

CORE_SIGNAL_COLUMNS = [
    "SPEC_PROCESSED_NB",
    "SPEC_RECEIVED_NB",
    "INF_A",
    "INF_B",
    "INF_ALL",
    "INF_NEGATIVE",
    "positivity_rate",
    "A_rate",
    "B_rate",
]

REGRESSION_TARGETS = [
    "target_next_INF_A",
    "target_next_INF_B",
    "target_next_INF_ALL",
    "target_next_positivity_rate",
    "target_next_A_rate",
    "target_next_B_rate",
]

CLASSIFICATION_TARGETS = [
    "target_trend",
    "target_subtype_driver",
    "target_increase_binary",
]

RANDOM_STATE = 42
