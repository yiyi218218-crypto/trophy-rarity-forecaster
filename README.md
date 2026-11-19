# trophy-rarity-forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yiyi218218-crypto/trophy-rarity-forecaster/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yiyi218218-crypto/trophy-rarity-forecaster/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Offline logit-scale quantile forecaster for GTA VI trophy earn rates, built for make.fun's launch-week forecasting bounty.

## The model

Each trophy's printed rate is read through the logit and fitted against `log(hours since release)`.
Consecutive identical reads are collapsed into one plateau per Exophase refresh, so a page that
repeats for six reads does not pretend to be six observations. The line is fitted with
Huber-reweighted least squares and a 19-hour recency half-life, then its slope is shrunk toward the
median slope of trophies in the same rarity band — at two decimals, one trophy's slope is noise.

The line is read at the target, damped where the rate saturates and where the slope is negative, and
turned into a normal quantile fan on the logit scale. The fan's width adds three independent terms:
the slope's standard error propagated over the horizon, a random walk in `sqrt(days)`, and the
binomial noise of a small earners count. When the page never refreshed once in the whole history,
the pending refresh's dilution is priced in explicitly, and a trophy already at the top gets a fan
pinned near 100 instead of one that spends logits on a move it cannot make.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, deterministic, one `<id>.out` for every `<id>.in`, and any failure inside a
case falls back to persistence so the output is always valid.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
