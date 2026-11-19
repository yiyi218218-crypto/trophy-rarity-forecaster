"""GTA VI trophy rarity: an offline logit-scale quantile forecaster.

Reads every <id>.in case in the input directory, writes <id>.out with five quantiles
(5/25/50/75/95) of every trophy's Exophase earn rate at the case's target.

The model
---------
A trophy's printed rate, through the logit, is a smooth function of the log of elapsed
hours since release.  Per trophy we collapse the page's plateaus (Exophase refreshes on
its own clock, so consecutive reads repeat), fit a Huber-weighted line over the recent
window, and shrink that slope toward the median slope of trophies in the same rarity
band -- a single trophy at two decimals is a noisy slope.  The line is read at the
target, damped where the rate saturates and where the slope is negative (launch-week
denominator growth does not extrapolate linearly), and turned into a normal quantile
fan on the logit scale.  The fan's width combines the slope's standard error, a random
walk in sqrt(days), and the binomial noise of a small earners count.  When the page
never refreshed once in the whole history, the pending refresh's dilution is priced in.

Standard library only; deterministic; every failure falls back to persistence.
"""

import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------- constants

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
NORMAL_Z = (-1.64485362695147, -0.67448975019608, 0.0, 0.67448975019608, 1.64485362695147)

LOOKBACK_HOURS = 54.0        # reads older than this do not steer the slope
RECENCY_HALF_LIFE = 19.0     # a read this old counts half as much as the newest
HUBER_C = 1.35               # outlier cut, in robust standard deviations
HUBER_PASSES = 3
RATE_FLOOR, RATE_CEIL = 0.01, 99.99
SATURATION_LOGIT = 0.95      # ~72%: growth starts to compress above here
SATURATION_SPAN = 2.6
RATE_TOP_PCT = 99.6
SLOPE_FLOOR, SLOPE_CEIL = -0.18, 2.30
DECLINE_ATTENUATION = 0.34   # a falling rate is not extrapolated at full speed

SHRINK_MIN, SHRINK_MAX = 0.18, 0.88
PRIOR_VAR_FLOOR = 0.010
LEVEL_BLEND = 0.72           # fitted level vs the raw last read

BASE_SD = 0.40
WALK_SD_PER_SQRT_DAY = 0.21
COUNT_SD_SCALE = 1.15
SD_FLOOR, SD_CEIL = 0.29, 1.15

# A page that never refreshed across the whole history is about to: on launch night the
# tracked base is a sliver of the players who will be counted an hour later, and every
# rate but the 100% one drops.  This is a structural prior, not a fitted number.
STALE_SPAN_HOURS = 12.0
STALE_SHIFT = -1.42
STALE_SD = 0.62

BANDS = ((0.8, 0), (1.5, 1), (2.5, 2), (5.0, 3), (15.0, 4), (1e9, 5))

# A trophy already at the top has nowhere to grow and cannot fall far: the tutorial
# trophy sat at 100.00 through launch night's refresh and is still the first thing every
# player earns.  A generic fan would spend two logits of pinball on it every day.
HIGH_RATE_GATE = 99.5
HIGH_RATE_FAN = (99.6, 99.85, 99.95, 99.98, 99.99)


# ---------------------------------------------------------------- small helpers

def clamp_rate(value):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return min(RATE_CEIL, max(RATE_FLOOR, v))


def to_logit(pct):
    p = min(RATE_CEIL, max(RATE_FLOOR, float(pct)))
    return math.log(p / (100.0 - p))


def from_logit(z):
    z = min(14.0, max(-14.0, float(z)))
    return 100.0 / (1.0 + math.exp(-z))


def parse_iso(text):
    if not isinstance(text, str) or not text.strip():
        return None
    raw = text.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def band_of(pct):
    for edge, tag in BANDS:
        if pct < edge:
            return tag
    return 5


def median_of(values, default):
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(clean) if clean else default


def variance_of(values, default):
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if len(clean) < 2:
        return default
    return statistics.pvariance(clean)


# ---------------------------------------------------------------- fitting

def collapse_plateaus(points):
    """One point per run of identical printed rates, stamped at the run's last read."""
    plateaus = []
    for hours, pct in points:
        if plateaus and abs(plateaus[-1][1] - pct) <= 1e-9:
            plateaus[-1][0] = hours
        else:
            plateaus.append([hours, pct])
    return [(h, p) for h, p in plateaus]


def weighted_line(xs, ys, ws):
    """Weighted least squares.  Returns the slope's standard error too: the sandwich form
    matters here, because normalising the weights away would hide the sample size and make
    every slope look ten times less certain than it is."""
    total = sum(ws)
    if total <= 0.0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / total
    my = sum(w * y for w, y in zip(ws, ys)) / total
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(1, len(xs) - 2)
    ssr = sum(w * r * r for w, r in zip(ws, resid))
    sigma2 = ssr * len(xs) / (total * dof)
    lever = sum(w * w * (x - mx) ** 2 for w, x in zip(ws, xs))
    var_slope = sigma2 * lever / (sxx * sxx)
    return intercept, slope, sxx, ws, resid, math.sqrt(max(var_slope, 1e-10))


def robust_trend(points, ref_hours):
    """Huber-reweighted least squares of logit(rate) on log(elapsed hours)."""
    if len(points) < 2:
        return None
    xs = [math.log(max(0.08, h)) for h, _ in points]
    ys = [to_logit(p) for _, p in points]
    ws = [0.5 ** (max(0.0, ref_hours - h) / RECENCY_HALF_LIFE) for h, _ in points]
    fit = None
    for _ in range(HUBER_PASSES):
        fit = weighted_line(xs, ys, ws)
        if fit is None:
            return None
        resid = fit[4]
        centre = median_of(resid, 0.0)
        spread = median_of([abs(r - centre) for r in resid], 0.0) * 1.4826
        spread = max(spread, 0.06)
        for k, r in enumerate(resid):
            over = abs(r)
            if over > HUBER_C * spread:
                ws[k] *= (HUBER_C * spread) / over
                if ws[k] < 1e-6:
                    ws[k] = 1e-6
    intercept, slope, _, _, _, se_slope = fit
    return {
        "intercept": intercept,
        "slope": slope,
        "x_last": xs[-1],
        "level": intercept + slope * xs[-1],
        "se_slope": min(0.80, se_slope),
    }


# ---------------------------------------------------------------- model

def forecast_case(case):
    trophies = case.get("trophies") or []
    reads = case.get("history") or []
    n = len(trophies)
    if n == 0:
        return {"trophies": []}

    game = case.get("game") or {}
    date_text = game.get("release_date")
    origin = None
    if isinstance(date_text, str):
        origin = parse_iso(date_text if "T" in date_text else date_text + "T00:00:00Z")
    if origin is None:
        origin = parse_iso(case.get("at"))
    if origin is None:
        origin = datetime(2026, 11, 19, tzinfo=timezone.utc)

    stamps = [parse_iso(read.get("at")) or parse_iso(read.get("slot")) for read in reads]
    known = [s for s in stamps if s is not None]
    first_stamp = min(known) if known else origin

    def elapsed(stamp):
        if stamp is None:
            stamp = first_stamp
        return max(0.05, (stamp - origin).total_seconds() / 3600.0 + 1.0)

    hours = [elapsed(s) for s in stamps]
    ref_hours = max(hours) if hours else 1.0
    horizon_days = max(0.02, float(case.get("horizon_minutes") or 60.0) / 1440.0)

    # Per-trophy observations, in read order.
    observed = []
    for idx in range(n):
        points = []
        for j, read in enumerate(reads):
            rates = read.get("rates") or []
            if idx < len(rates):
                pct = clamp_rate(rates[idx])
                if pct is not None:
                    points.append((hours[j], pct))
        observed.append(points)

    # Fits over the recent window, one per trophy.
    fits = []
    for points in observed:
        if not points:
            fits.append(None)
            continue
        plats = collapse_plateaus(points)
        recent = [p for p in plats if p[0] >= ref_hours - LOOKBACK_HOURS]
        if len(recent) < 2:
            recent = plats[-3:]
        fits.append(robust_trend(recent, ref_hours))

    last_pct = [points[-1][1] if points else None for points in observed]
    bands = [band_of(p) if p is not None else 1 for p in last_pct]

    # Slope priors by rarity band, medians of the other members.
    slopes = [f["slope"] if f else None for f in fits]
    priors = []
    for idx in range(n):
        peers = [slopes[j] for j in range(n) if j != idx and bands[j] == bands[idx]
                 and slopes[j] is not None]
        if len(peers) < 4:
            peers = [slopes[j] for j in range(n) if j != idx and slopes[j] is not None]
        priors.append(median_of(peers, 0.0))
    prior_var = max(PRIOR_VAR_FLOOR, variance_of([s for s in slopes if s is not None], PRIOR_VAR_FLOOR))

    # Is the whole page one frozen plateau?  Then a refresh is pending and dilutes.
    frozen = False
    if reads:
        span = ref_hours - min(hours)
        tracking = {read.get("players_tracked") for read in reads}
        frozen = (span <= STALE_SPAN_HOURS and len(tracking) == 1
                  and all(len(collapse_plateaus(p)) <= 1 for p in observed))

    players_last = 0.0
    for read in reversed(reads):
        value = read.get("players_tracked")
        if isinstance(value, (int, float)) and value > 0:
            players_last = float(value)
            break

    target_hours = ref_hours + horizon_days * 24.0
    x_target = math.log(max(0.08, target_hours))
    x_ref = math.log(max(0.08, ref_hours))

    rows = []
    for idx in range(n):
        pct = last_pct[idx]
        fit = fits[idx]
        if pct is None:
            pct = 0.63
        if pct >= HIGH_RATE_GATE:
            rows.append({"id": trophies[idx].get("id", idx + 1), "q": list(HIGH_RATE_FAN)})
            continue
        if fit is None:
            level = to_logit(pct)
            slope_used = 0.0
            se_delta = 0.30
        else:
            var_b = fit["se_slope"] ** 2 + 1e-6
            weight = min(SHRINK_MAX, max(SHRINK_MIN, prior_var / (prior_var + var_b)))
            slope_used = weight * fit["slope"] + (1.0 - weight) * priors[idx]
            slope_used = min(SLOPE_CEIL, max(SLOPE_FLOOR, slope_used))
            if slope_used < 0.0:
                slope_used *= DECLINE_ATTENUATION
            level = LEVEL_BLEND * fit["level"] + (1.0 - LEVEL_BLEND) * to_logit(pct)
            se_delta = fit["se_slope"] * abs(x_target - fit["x_last"]) + 0.05

        step = slope_used * max(0.0, x_target - x_ref)
        if level > SATURATION_LOGIT:
            compress = 1.0 - (level - SATURATION_LOGIT) / SATURATION_SPAN
            step *= max(0.15, compress)
        centre = level + step
        centre = min(to_logit(RATE_TOP_PCT), max(to_logit(0.02), centre))

        if frozen and pct < 99.0:
            centre = min(to_logit(RATE_TOP_PCT), max(to_logit(0.02), centre + STALE_SHIFT))

        earners = max(12.0, pct / 100.0 * (players_last if players_last > 0 else 500.0))
        spread = math.sqrt(
            BASE_SD ** 2
            + (WALK_SD_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
            + se_delta ** 2
            + (COUNT_SD_SCALE / math.sqrt(earners)) ** 2
        )
        if frozen and pct < 99.0:
            spread = math.hypot(spread, STALE_SD)
        spread = min(SD_CEIL, max(SD_FLOOR, spread))

        quantiles = [round(from_logit(centre + z * spread), 4) for z in NORMAL_Z]
        quantiles = [min(100.0, max(0.0, q)) for q in quantiles]
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophies[idx].get("id", idx + 1), "q": quantiles})

    return {"trophies": rows}


def persistence_row(trophy_id, pct, horizon_days):
    centre = to_logit(pct)
    spread = min(SD_CEIL, max(SD_FLOOR, 0.5 * math.sqrt(max(0.02, horizon_days))))
    return {"id": trophy_id,
            "q": [round(from_logit(centre + z * spread), 4) for z in NORMAL_Z]}


def fallback_output(case):
    days = max(0.02, float(case.get("horizon_minutes") or 60.0) / 1440.0)
    rows = []
    trophies = case.get("trophies") or []
    reads = case.get("history") or []
    for idx, trophy in enumerate(trophies):
        pct = 0.63
        for read in reversed(reads):
            rates = read.get("rates") or []
            if idx < len(rates):
                value = clamp_rate(rates[idx])
                if value is not None:
                    pct = value
                    break
        rows.append(persistence_row(trophy.get("id", idx + 1), pct, days))
    return {"trophies": rows}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.glob("*.in")):
        case = None
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            payload = forecast_case(case)
        except Exception:
            try:
                payload = fallback_output(case) if case else {"trophies": []}
            except Exception:
                payload = {"trophies": []}
        text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        (output_dir / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
