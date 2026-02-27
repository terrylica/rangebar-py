"""Re-audit direction patterns with corrected Hurst assumption.

Issue #57 / Task #150: The Hurst re-audit found H~0.58 (near random walk),
not H~0.79 as originally claimed.

Re-audit direction patterns (U/D, 2-bar, 3-bar) with:
1. Corrected effective sample size calculation (H=0.58)
2. Verify return distribution is actually at boundary
3. Re-compute t-statistics with proper temporal-safe shifts

Query data from bigblack ClickHouse.
"""

import json
from datetime import UTC, datetime

import numpy as np


def log(level: str, message: str, **kwargs: object) -> None:
    """Log in NDJSON format."""
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level,
        "message": message,
        **kwargs,
    }
    print(json.dumps(entry), flush=True)


def compute_t_statistic(
    p_observed: float, p_null: float, n_samples: int, h: float = 0.5
) -> tuple[float, float]:
    """Compute t-statistic with Hurst-adjusted effective sample size.

    Args:
        p_observed: Observed proportion
        p_null: Null hypothesis proportion
        n_samples: Raw sample size
        h: Hurst exponent (H=0.5 for random walk)

    Returns:
        (t_stat_raw, t_stat_adjusted) - raw and Hurst-adjusted t-statistics
    """
    # Standard error under null
    se_raw = np.sqrt(p_null * (1 - p_null) / n_samples)
    t_raw = (p_observed - p_null) / se_raw

    # Hurst-adjusted effective sample size
    # T_eff = T^(2*(1-H)) for H > 0.5
    n_eff = n_samples ** (2 * (1 - h)) if h != 0.5 else n_samples

    se_adj = np.sqrt(p_null * (1 - p_null) / n_eff)
    t_adj = (p_observed - p_null) / se_adj

    return t_raw, t_adj


def main() -> None:
    """Re-audit direction patterns."""
    log("INFO", "Starting direction pattern re-audit")

    import clickhouse_connect
    from rangebar.clickhouse.tunnel import SSHTunnel

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    threshold = 100

    # Hurst exponent from multi-estimator audit
    h_corrected = 0.578  # Mean of R/S, DFA, Variance-Time

    log("INFO", "Using corrected Hurst exponent", H=h_corrected)

    with SSHTunnel("bigblack") as local_port:
        client = clickhouse_connect.get_client(
            host="localhost",
            port=local_port,
            database="rangebar_cache",
        )

        all_results = []

        for symbol in symbols:
            log("INFO", "Analyzing symbol", symbol=symbol)

            # Query direction sequence with temporal-safe shifts
            # Features: shift(1) = previous bar
            # Target: shift(-1) = next bar
            query = f"""
            SELECT
                close_time_ms,
                CASE WHEN close > open THEN 'U' ELSE 'D' END AS direction,
                lagInFrame(CASE WHEN close > open THEN 'U' ELSE 'D' END, 1)
                    OVER (ORDER BY close_time_ms) AS prev_direction,
                leadInFrame(CASE WHEN close > open THEN 'U' ELSE 'D' END, 1)
                    OVER (ORDER BY close_time_ms) AS next_direction,
                (close - open) / open * 10000 AS return_dbps,
                (high - low) / low * 10000 AS range_dbps
            FROM range_bars
            WHERE symbol = '{symbol}'
              AND threshold_decimal_bps = {threshold}
              AND ouroboros_mode = 'year'
            ORDER BY close_time_ms
            """

            result = client.query(query)
            rows = result.result_rows

            log("INFO", "Data loaded", symbol=symbol, n_bars=len(rows))

            # Extract data
            directions = [r[1] for r in rows]
            prev_directions = [r[2] for r in rows]
            next_directions = [r[3] for r in rows]
            returns = np.array([r[4] for r in rows], dtype=np.float64)
            ranges = np.array([r[5] for r in rows], dtype=np.float64)

            # 1. Check return distribution
            at_threshold_pct = (
                np.sum(np.abs(np.abs(returns) - threshold) < 5) / len(returns) * 100
            )
            log(
                "INFO",
                "Return distribution check",
                symbol=symbol,
                pct_within_5dbps_of_threshold=round(at_threshold_pct, 1),
                mean_return=round(np.mean(returns), 2),
                std_return=round(np.std(returns), 2),
                mean_range=round(np.mean(ranges), 2),
            )

            # 2. Single-bar direction autocorrelation (temporal-safe)
            # P(next_direction = U | direction = U)
            valid_mask = [
                i
                for i in range(len(rows))
                if prev_directions[i] is not None and next_directions[i] is not None
            ]

            u_bars = [i for i in valid_mask if directions[i] == "U"]
            d_bars = [i for i in valid_mask if directions[i] == "D"]

            # P(next = U | current = U)
            p_uu = sum(1 for i in u_bars if next_directions[i] == "U") / len(u_bars)
            # P(next = D | current = D)
            p_dd = sum(1 for i in d_bars if next_directions[i] == "D") / len(d_bars)

            # Null hypothesis: P(next = U) = P(U) overall
            p_u_null = sum(1 for d in directions if d == "U") / len(directions)
            p_d_null = 1 - p_u_null

            t_uu_raw, t_uu_adj = compute_t_statistic(p_uu, p_u_null, len(u_bars), h_corrected)
            t_dd_raw, t_dd_adj = compute_t_statistic(p_dd, p_d_null, len(d_bars), h_corrected)

            log(
                "INFO",
                "Single-bar autocorrelation",
                symbol=symbol,
                P_UU=round(p_uu, 4),
                P_DD=round(p_dd, 4),
                P_U_null=round(p_u_null, 4),
                t_UU_raw=round(t_uu_raw, 2),
                t_UU_adjusted=round(t_uu_adj, 2),
                t_DD_raw=round(t_dd_raw, 2),
                t_DD_adjusted=round(t_dd_adj, 2),
                n_U_bars=len(u_bars),
                n_D_bars=len(d_bars),
            )

            # 3. 2-bar patterns (UU, UD, DU, DD) → next direction
            patterns_2bar = {}
            for i in valid_mask:
                prev_d = prev_directions[i]
                next_d = next_directions[i]
                if not prev_d or not next_d or prev_d not in ("U", "D") or next_d not in ("U", "D"):
                    continue
                pattern = prev_d + directions[i]
                if pattern not in patterns_2bar:
                    patterns_2bar[pattern] = {"U": 0, "D": 0}
                patterns_2bar[pattern][next_d] += 1

            for pattern, counts in sorted(patterns_2bar.items()):
                total = counts["U"] + counts["D"]
                if total < 100:
                    continue
                p_u_after = counts["U"] / total
                t_raw, t_adj = compute_t_statistic(p_u_after, p_u_null, total, h_corrected)

                log(
                    "INFO",
                    "2-bar pattern",
                    symbol=symbol,
                    pattern=pattern,
                    P_U_next=round(p_u_after, 4),
                    n_samples=total,
                    t_raw=round(t_raw, 2),
                    t_adjusted=round(t_adj, 2),
                    significant_raw=bool(abs(t_raw) >= 3.0),
                    significant_adj=bool(abs(t_adj) >= 3.0),
                )

                all_results.append(
                    {
                        "symbol": symbol,
                        "pattern": pattern,
                        "p_u_next": p_u_after,
                        "n_samples": total,
                        "t_raw": t_raw,
                        "t_adjusted": t_adj,
                    }
                )

        # Summary
        log("INFO", "Re-audit complete", total_patterns=len(all_results))

        # Check for ODD robust patterns (significant across ALL symbols)
        pattern_groups = {}
        for r in all_results:
            p = r["pattern"]
            if p not in pattern_groups:
                pattern_groups[p] = []
            pattern_groups[p].append(r)

        odd_robust = []
        for pattern, results in pattern_groups.items():
            if len(results) < 4:  # Need all 4 symbols
                continue

            # Check if all have same sign t-stat and |t| >= 3.0
            signs = [1 if r["t_adjusted"] > 0 else -1 for r in results]
            all_same_sign = len(set(signs)) == 1
            all_significant = all(abs(r["t_adjusted"]) >= 3.0 for r in results)

            if all_same_sign and all_significant:
                odd_robust.append(
                    {
                        "pattern": pattern,
                        "min_t": min(abs(r["t_adjusted"]) for r in results),
                        "direction": "continuation" if signs[0] > 0 else "reversal",
                    }
                )

        if odd_robust:
            log(
                "INFO",
                "ODD ROBUST patterns found with corrected Hurst!",
                n_patterns=len(odd_robust),
                patterns=odd_robust,
            )
        else:
            log(
                "INFO",
                "NO ODD robust patterns even with corrected Hurst",
                verdict="Direction pattern invalidation CONFIRMED",
            )


if __name__ == "__main__":
    main()
