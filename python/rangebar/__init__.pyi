"""Type stubs for rangebar package.

Public API re-exports. Per-module stubs live alongside their source files.
See PEP 561 and https://typing.python.org/en/latest/guides/writing_stubs.html
"""

# Version
__version__: str

# Exceptions (rangebar.exceptions, rangebar.validation.continuity)
# Alpha-forge compatibility layer (rangebar.compat) — Issue #95
# Binance Vision probe (rangebar.binance_vision)
from .binance_vision import (
    BINANCE_VISION_AGGTRADES_URL as BINANCE_VISION_AGGTRADES_URL,
)
from .binance_vision import (
    probe_latest_available_date as probe_latest_available_date,
)

# Cache population (rangebar.checkpoint)
from .checkpoint import populate_cache_resumable as populate_cache_resumable
from .compat import get_available_symbols as get_available_symbols
from .compat import get_cache_coverage as get_cache_coverage
from .compat import get_feature_groups as get_feature_groups
from .compat import get_feature_manifest as get_feature_manifest
from .compat import get_range_bars_panel as get_range_bars_panel
from .compat import to_panel_format as to_panel_format

# Constants (rangebar.constants)
from .constants import BAR_FLAG_COLUMNS as BAR_FLAG_COLUMNS
from .constants import INTER_BAR_FEATURE_COLUMNS as INTER_BAR_FEATURE_COLUMNS
from .constants import LONG_RANGE_DAYS as LONG_RANGE_DAYS
from .constants import THRESHOLD_DECIMAL_MAX as THRESHOLD_DECIMAL_MAX
from .constants import THRESHOLD_DECIMAL_MIN as THRESHOLD_DECIMAL_MIN
from .constants import THRESHOLD_PRESETS as THRESHOLD_PRESETS
from .constants import TIER1_SYMBOLS as TIER1_SYMBOLS
from .constants import TRADE_ID_RANGE_COLUMNS as TRADE_ID_RANGE_COLUMNS
from .exceptions import SymbolNotRegisteredError as SymbolNotRegisteredError

# Main API (rangebar.orchestration.*)
from .orchestration.count_bounded import get_n_range_bars as get_n_range_bars

# Orchestration models (rangebar.orchestration.models)
from .orchestration.models import PrecomputeProgress as PrecomputeProgress
from .orchestration.models import PrecomputeResult as PrecomputeResult
from .orchestration.precompute import precompute_range_bars as precompute_range_bars
from .orchestration.range_bars import get_range_bars as get_range_bars
from .orchestration.range_bars import get_range_bars_pandas as get_range_bars_pandas

# Ouroboros (rangebar.ouroboros)
from .ouroboros import OrphanedBarMetadata as OrphanedBarMetadata
from .ouroboros import OuroborosBoundary as OuroborosBoundary
from .ouroboros import OuroborosMode as OuroborosMode
from .ouroboros import get_ouroboros_boundaries as get_ouroboros_boundaries

# Trade processing (rangebar.processors.api)
from .processors.api import process_trades_polars as process_trades_polars
from .processors.api import process_trades_polars_lazy as process_trades_polars_lazy
from .processors.api import (
    process_trades_to_dataframe as process_trades_to_dataframe,
)

# Recency backfill (rangebar.recency)
from .recency import BackfillResult as BackfillResult
from .recency import LoopState as LoopState
from .recency import backfill_all_recent as backfill_all_recent
from .recency import backfill_recent as backfill_recent
from .recency import run_adaptive_loop as run_adaptive_loop

# Streaming sidecar (rangebar.sidecar) — Issue #91
from .sidecar import SidecarConfig as SidecarConfig
from .sidecar import run_sidecar as run_sidecar

# Symbol registry (rangebar.symbol_registry)
from .symbol_registry import SymbolEntry as SymbolEntry
from .symbol_registry import SymbolTransition as SymbolTransition
from .symbol_registry import (
    clear_symbol_registry_cache as clear_symbol_registry_cache,
)
from .symbol_registry import get_effective_start_date as get_effective_start_date
from .symbol_registry import get_registered_symbols as get_registered_symbols
from .symbol_registry import get_symbol_entries as get_symbol_entries
from .symbol_registry import get_transitions as get_transitions
from .symbol_registry import (
    validate_and_clamp_start_date as validate_and_clamp_start_date,
)
from .symbol_registry import validate_symbol_registered as validate_symbol_registered

# Tiered validation (rangebar.validation.continuity)
from .validation.continuity import ASSET_CLASS_MULTIPLIERS as ASSET_CLASS_MULTIPLIERS
from .validation.continuity import VALIDATION_PRESETS as VALIDATION_PRESETS
from .validation.continuity import AssetClass as AssetClass
from .validation.continuity import ContinuityError as ContinuityError
from .validation.continuity import ContinuityWarning as ContinuityWarning
from .validation.continuity import GapInfo as GapInfo
from .validation.continuity import GapTier as GapTier
from .validation.continuity import TieredValidationResult as TieredValidationResult
from .validation.continuity import TierSummary as TierSummary
from .validation.continuity import TierThresholds as TierThresholds
from .validation.continuity import ValidationPreset as ValidationPreset
from .validation.continuity import detect_asset_class as detect_asset_class
from .validation.continuity import (
    validate_continuity_tiered as validate_continuity_tiered,
)
