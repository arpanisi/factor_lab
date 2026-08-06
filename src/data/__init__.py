"""Data adapters for Factor Lab namespaces."""

from src.data.adapters import adapt_crypto_ohlcv, adapt_crsp_dsf_v2, adapt_taq_features

# wrds_crsp.py / wrds_taq.py are intentionally not re-exported here: they use
# bare same-directory imports (e.g. `from adapters import ...`) so they stay
# runnable directly as `python src/data/wrds_crsp.py`. Importing them through
# this package would execute those bare imports under a different sys.path
# and fail. Import them directly from their file if you need their functions.

__all__ = [
    "adapt_crypto_ohlcv",
    "adapt_crsp_dsf_v2",
    "adapt_taq_features",
]
