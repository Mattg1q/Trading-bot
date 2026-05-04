import os
import sys

def _load_env_file(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_load_env_file()

# Binance Futures Testnet Configuration
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

# Finnhub News Configuration
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# Trading Parameters
TICKER = "BTC/USDT"
BINANCE_STREAM_SYMBOL = "btcusdt"
LEVERAGE = int(os.getenv("LEVERAGE", "3"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "1")) # seconds
DEFAULT_CAPITAL = float(os.getenv("DEFAULT_CAPITAL", "5000.0"))
LOB_LEVELS = int(os.getenv("LOB_LEVELS", "20"))
LOB_STREAM_SPEED = os.getenv("LOB_STREAM_SPEED", "100ms")
FINNHUB_MAX_HEADLINES = int(os.getenv("FINNHUB_MAX_HEADLINES", "5"))
FINNHUB_NEWS_TTL_SECONDS = int(os.getenv("FINNHUB_NEWS_TTL_SECONDS", "300"))
SNN_WEIGHTS_PATH = os.getenv("SNN_WEIGHTS_PATH", "models/snn_weights.pth")
ALLOW_UNTRAINED_SNN = os.getenv("ALLOW_UNTRAINED_SNN", "0") == "1"
REQUIRE_GPU = os.getenv("REQUIRE_GPU", "1") == "1"
MIN_PYTHON = (3, 11)
MAX_PYTHON = (3, 11)

# Mode: "Paper Trading" or "Demo Futures"
TRADING_MODE = os.getenv("TRADING_MODE", "Demo Futures")

def validate_python_runtime():
    if sys.version_info[:2] != MIN_PYTHON:
        raise RuntimeError(
            f"Python 3.11 is required, but this environment is "
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}."
        )

def validate_required_secrets():
    missing = []
    if TRADING_MODE == "Demo Futures":
        if not BINANCE_API_KEY:
            missing.append("BINANCE_API_KEY")
        if not BINANCE_SECRET_KEY:
            missing.append("BINANCE_SECRET_KEY")
    if not FINNHUB_API_KEY:
        missing.append("FINNHUB_API_KEY")
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )
