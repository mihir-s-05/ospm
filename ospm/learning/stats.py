"""Paired significance for system comparisons on the same tasks (as in Fan et al. 2026, harness design study).

A pair of systems differs on a task when one scores higher; ties carry no information. The exact McNemar test is
the two-sided binomial test of the discordant pairs (wins vs losses) at p = 0.5. Comparisons of several systems
against one baseline are corrected together with Benjamini-Hochberg.
"""

from __future__ import annotations

from math import comb


def mcnemar(wins: int, losses: int) -> float:
    """Exact two-sided McNemar p-value from the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def bh(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    out, running = [1.0] * m, 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * m / rank)
        out[i] = min(1.0, running)
    return out


def paired(rows: dict[str, list[dict]], base: str, key: str = "score") -> dict[str, dict]:
    """{system: {wins, losses, ties, p, p_bh}} for every system against `base` over aligned rows."""
    out = {}
    for s, v in rows.items():
        if s == base:
            continue
        wins = sum(1 for x, y in zip(v, rows[base]) if x[key] > y[key] + 1e-9)
        losses = sum(1 for x, y in zip(v, rows[base]) if x[key] < y[key] - 1e-9)
        out[s] = {"wins": wins, "losses": losses, "ties": min(len(v), len(rows[base])) - wins - losses,
                  "p": round(mcnemar(wins, losses), 4)}
    for s, q in zip(out, bh([d["p"] for d in out.values()])):
        out[s]["p_bh"] = round(q, 4)
    return out
