# Issue #46: Modularization - ClickHouse-level constants
"""ClickHouse cache layer constants.

SSoT for ClickHouse-specific settings shared across mixins.
"""

# Issue #90: Parallelize FINAL dedup across partitions for ~7x faster reads.
# Safe because our partitions (symbol, threshold, month) are independent.
FINAL_READ_SETTINGS: dict[str, int] = {
    "do_not_merge_across_partitions_select_final": 1,
}
