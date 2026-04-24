# Strategy

## What We Built

This bot is a two-leg round-2 Prosperity strategy built around a simple split:

- `ASH_COATED_OSMIUM` is treated as the stable leg.
- `INTARIAN_PEPPER_ROOT` is treated as the directional leg.

That split matters because the best edge is not one universal model. It is a product-specific decomposition:

- Pepper wants early directional conviction, trend recognition, and fast reaction when orderflow turns.
- Osmium wants mean reversion, inventory skew, and disciplined market making.

## The Logical Process

We did not start with a giant monolithic model. The process was more incremental:

1. Keep the mandatory constraints intact.
2. Test aggressive variants and reject the ones that hurt PnL.
3. Add regime logic only where it changed actual execution.
4. Treat the two assets differently instead of forcing one framework onto both.
5. Keep every useful signal small, bounded, and persistent across ticks.

That process produced a system that is mostly a layered signal combiner, not a single magic indicator.

## Pepper Root Logic

Pepper is where the directional alpha lives.

The model uses:

- fast and slow EMAs for short-vs-long trend
- top-of-book imbalance
- deep imbalance from multiple levels of the book
- a Kalman-style fair value and uncertainty model
- a rolling short-window trend bias
- a professor-style orderflow overlay built from recent price movement and MA distance
- wavelet chop as a regime scaler for spreads and taker thresholds
- correlation-based hedge awareness from Osmium

The goal is not to predict every tick. The goal is to align with short-horizon drift when it is real and to back off when the tape is noisy.

### Pepper Execution Style

Pepper uses three modes of action:

- passive quoting when the signal is moderate
- aggressive taking when drift is strong enough to pay for the spread
- small OBI scalps when the book is giving a short-lived edge

The bot also tapers risk in the warmup window and near the close.

## Osmium Logic

Osmium is the stabilizer and maker leg.

The model uses:

- mid EMA and book EMA
- slope EMA for trend detection
- inventory skew for reservation price control
- a short rolling mid-window for mean reversion context
- OBI-smoothed reversion bias
- dip-buy and rebound-sell overlays
- optional sweep-taking when the book is clearly mispriced

This leg is intentionally more conservative. It should harvest spread and reversion without becoming the source of large inventory blowups.

### Osmium Regime Thinking

The professor-style note we applied maps well here:

- moving-average distance helps identify reversion opportunities
- cumulative orderbook imbalance helps detect pressure building into reversals

That is why Osmium now leans on both slope/trend and MA-distance mean reversion.

## Regime Detection

We use a lightweight wavelet chop estimate rather than a heavy regime model.

Why:

- raw price swings can be misleading
- log-return energy is scale-aware
- a short Haar transform is cheap enough to run every tick
- the output is easy to interpret: high chop vs low chop

This regime signal does not replace the core strategy. It shapes how fast the strategy should react.

## Risk Management

The bot stays alive by being restrictive in the right places:

- hard 80-unit position limits
- warmup throttling
- taker thresholds
- volatility-based sizing
- drawdown-based scaling
- correlation hedging
- inventory skew

That combination is more useful than a single stop-loss rule.

## Why The Earlier Aggressive Idea Failed

We tested more aggressive variants and found that removing too many filters usually reduced PnL rather than improving it.

The reason is simple:

- the book is noisy
- the spread costs matter
- some signals are only useful when gated

So the winning direction was not maximum aggression. It was selective aggression with regime awareness.

## What Actually Worked

The most useful pieces were:

- Pepper trend and orderflow alignment
- Osmium inventory-skewed mean reversion
- wavelet chop as a quote/take modifier
- correlation awareness across the two legs
- bounded, reusable state across ticks

## Current Model Results

The current deployed version reached:

- Rust round 2 total: `255920`
- Prosperity4btx total: `256156`

That is the reference result set for this snapshot.

## Extra Notes

A few design choices are worth keeping in mind:

- NumPy is used where repeated reductions are worth vectorizing.
- pandas is used sparingly for bounded rolling means, not as a general dependency.
- The strategy is intentionally modular so one leg can be tuned without wrecking the other.
- The bid requirement is still fixed at `2141`.

## Final Take

The real edge is not one exotic indicator. It is the combination of:

- correct product decomposition
- selective use of trend and reversion
- persistence of state
- disciplined execution thresholds
- continuous backtest validation
