# Code Documentation

This document walks through the current trading system from top to bottom and captures the behavior, state, and observed backtest outcomes.

## 1. File Purpose

The active trading system is implemented in `trader.py`. It is a two-asset round-2 trader for IMC Prosperity 4 with these responsibilities:

- Trade `ASH_COATED_OSMIUM` as a mean-reverting market-making product with inventory awareness.
- Trade `INTARIAN_PEPPER_ROOT` as a directional, signal-driven product with trend, orderflow, and regime logic.
- Preserve state across ticks through `traderData` JSON.
- Keep the hard `bid()` requirement returning `2141`.
- Respect the 80-unit hard position limit on both assets.

This document reflects the current version of the system after the wavelet, Kalman, correlation, and efficiency passes.

## 2. Imports and Dependencies

The file imports the Prosperity model types from `datamodel.py`:

- `Listing`
- `Observation`
- `Order`
- `OrderDepth`
- `ProsperityEncoder`
- `Symbol`
- `Trade`
- `TradingState`

It also uses:

- `math` for rounding and spread calculations.
- `json` for persistent state serialization.
- `typing.Any` and `typing.Optional` for helpers.
- `numpy` for fast vector-style reductions over book data and rolling windows.
- `pandas` for the short bounded rolling mean helper.

The NumPy and pandas usage is limited to repeated reductions and short rolling-window operations. The trading logic itself still stays in plain Python and dictionary-based state.

## 3. Logger

The `Logger` class is a Prosperity output compressor. It packs state, orders, conversions, trader data, and logs into the JSON shape expected by the simulator.

Key methods:

- `print(...)` appends formatted text to the internal log buffer.
- `flush(...)` emits the compressed JSON payload and clears the buffer.
- `compress_state(...)` serializes the current trading state.
- `compress_listings(...)`, `compress_order_depths(...)`, `compress_trades(...)`, `compress_observations(...)`, and `compress_orders(...)` transform simulator objects into compact JSON-friendly lists and dictionaries.
- `truncate(...)` shortens large strings safely so the log payload fits within the simulator limit.

This class is infrastructure only. It does not affect strategy decisions.

## 4. Trader Class Overview

`Trader` is the active decision engine.

### 4.1 Hard Constraints

The class enforces the round constraints directly:

- `bid()` returns `2141` exactly.
- `POSITION_LIMITS` sets both products to `80`.
- The main execution loop sizes orders through `buy_cap` and `sell_cap` so it never intentionally exceeds the configured limit.

### 4.2 Execution Modes

The code currently has a single deployed behavior, but internally it still retains some mode flags and legacy toggles used during iteration:

- `AGGRESSIVE_EXPERIMENT` is disabled.
- `ENABLE_DYNAMIC_KALMAN` is enabled.
- `ENABLE_RISK_PARITY` is derived from the aggressive toggle.
- `ENABLE_AGGRESSIVE_OBI` is enabled.
- `ENABLE_CORRELATION_HEDGE` is derived from the aggressive toggle.

These toggles exist so the bot can switch risk posture without changing the core control flow.

## 5. Helper Methods

### 5.1 `_clamp`

A small utility for keeping values inside bounds. It is used throughout the code to keep spreads, thresholds, and signal weights stable.

### 5.2 `_append_iceberg`

This helper breaks a larger desired order into smaller child orders. It is used to reduce the chance of sending one large order when smaller chunks are preferred.

### 5.3 `_clone_default`

This deep-copies list and dict defaults when reconstructing persistent state.

### 5.4 `_window_mean`

This helper uses pandas to compute a bounded mean from a short history window. It is used in the Pepper branch for short-window trend calculations.

### 5.5 `_update_volatility_tracking`

This helper tracks a short rolling price window and computes log-return volatility with NumPy.

Behavior:

- Keeps the last 20 mid prices.
- Converts prices to log returns.
- Computes sample variance of returns.
- Stores the result under `volatility_values` in trader data.

This volatility estimate feeds the dynamic Kalman parameter selection and several risk controls.

### 5.6 `_update_wavelet_regime`

This estimates a chop/trend regime using a one-level Haar transform on log returns.

Behavior:

- Keeps a bounded wavelet window of mid prices.
- Converts to log returns.
- Applies a pairwise Haar split into approximation and detail terms.
- Computes detail energy divided by total energy.
- Smooths the regime score with an EMA.
- Stores the result under `wavelet_chop` and `wavelet_chop_ema`.

Interpretation:

- Higher values mean more chop.
- Lower values mean a cleaner directional tape.

### 5.7 `_get_dynamic_kalman_params`

This computes Kalman process noise `Q` and measurement noise `R` based on:

- volatility
- order book imbalance
- trend strength
- wavelet chop

High volatility and high chop widen noise assumptions. Trend strength tightens adaptation when the tape is directional.

### 5.8 `_calculate_var_95`

This computes a 95% VaR estimate from recent returns.

If there is not enough history, it returns a conservative default.

### 5.9 `_update_returns_buffer`

Adds realized returns to the rolling history used by the VaR helper.

### 5.10 `_calculate_risk_parity_weights`

Computes inverse-volatility weights for OSMIUM and PEPPER.

### 5.11 `_drawdown_scale`

Returns a sizing multiplier based on recent drawdown.

- Deep drawdown cuts sizing.
- Very calm periods slightly expand utilization.

### 5.12 `_estimate_correlation`

Tracks a rolling return correlation between OSMIUM and PEPPER.

This is used as a hedge signal when cross-asset moves become tightly linked.

### 5.13 `_dynamic_correlation_threshold`

Raises the hedge trigger threshold in more volatile conditions.

### 5.14 `_volatility_position_scale`

Reduces exposure when volatility rises. This is a risk-off scaler.

### 5.15 `_select_osmium_model`

Chooses between the Kalman and regime versions of OSMIUM.

Inputs used:

- volatility
- slope EMA
- wavelet chop
- distance from fair value

General behavior:

- In low-volatility, choppy conditions, it may prefer the regime branch.
- In trending or volatile conditions, it prefers Kalman.
- The branch is sticky enough to avoid constant switching.

## 6. Persistent State

The trader persists a large amount of state in `traderData`. This is central to the design because the engine is called tick-by-tick and otherwise forgets history.

Current state keys include:

- generic mid / EMA / return caches
- Pepper state: `pepper_level`, `pepper_velocity`, `pepper_var`, `pepper_prev_kalman_gain`, `pepper_innovation_ema`, `pepper_orderflow_ema`, `pepper_obi_ema`, `pepper_mid_window`
- Osmium state: `osmium_mid_ema`, `osmium_book_ema`, `osmium_slope_ema`, `osmium_mean`, `osmium_var`, `osmium_regime`, `osmium_mid_window`, `osmium_obi_ema`, `osmium_last_sweep_ts`, `osmium_active_model`
- risk and analytics state: `volatility_window`, `volatility_values`, `returns_buffer`, `correlation_buffers`, `correlation_osmium_pepper`, `pnl_osmium`, `pnl_pepper`, `wavelet_window`, `wavelet_chop`, `wavelet_chop_ema`

The restore path validates and repairs missing or malformed keys so the bot keeps running if the JSON changes format.

## 7. Main Run Loop

`run(self, state)` is the entry point each tick.

High-level flow:

1. Restore prior state from `state.traderData`.
2. Read current order books for both products.
3. Update rolling volatility and wavelet regime for each product.
4. Update cross-asset correlation when both books are available.
5. Compute per-product size scales from volatility.
6. Route OSMIUM to `_osmium(...)`.
7. Route PEPPER to `_pepper(...)`, passing the current OSMIUM position for hedge awareness.
8. Serialize the updated state back to `traderData`.
9. Flush logs and return the order dictionary, conversions, and trader data.

This structure means all the actual strategy decisions happen inside the product-specific branches, while the top-level loop is mostly orchestration.

## 8. OSMIUM Strategy

OSMIUM is the mean-reversion and market-making leg.

### 8.1 Branching

`_osmium(...)` dispatches to either `_osmium_kalman(...)` or `_osmium_regime(...)` depending on the model selector.

### 8.2 Kalman Branch

This is the more complex and currently favored branch.

Core components:

- Uses a product profile to tune trend cap, dip offsets, sweep gaps, inventory skew, and ladder weights.
- Builds a mid EMA and a book EMA.
- Computes slope EMA for trend detection.
- Applies inventory skew to fair value.
- Applies a short rolling mid-window mean-reversion bias using the professor-style note about MA distance.
- Uses a dip-buy overlay to buy only when the price, slope, and OBI all agree on a real dip.
- Uses a rebound-sell overlay to take profit as price reverts.
- Has a sweep-taker mode for larger imbalance or mispricing opportunities.
- Builds maker quotes around a reservation price with layered size.

Important design details:

- The strategy is not pure market making.
- It blends mean reversion, adaptive fair value, and inventory control.
- The sweep logic is time-gated early in the session.
- The ladder size changes with volatility and inventory direction.

### 8.3 Regime Branch

The regime branch is a lighter mean-reversion model:

- Maintains a mean and variance estimate.
- Tracks a regime score from return movement and OBI.
- Uses a mean reversion band based on variance.
- Trades swings when price deviates enough from the estimated fair.
- Still ends with maker quotes and inventory skew.

## 9. PEPPER Strategy

PEPPER is the directional leg and the highest-activity signal engine.

### 9.1 Core Signal Stack

The branch combines several layers:

- Fast EMA and slow EMA for short-vs-long trend.
- Top-of-book OBI.
- Deep OBI from the top 10 levels.
- A blended OBI that weights surface and depth pressure.
- A Kalman-style fair value estimate and measurement noise model.
- A mid-window trend bias using short and long rolling averages.
- A professor-style orderflow overlay based on short-horizon price movement, moving-average distance, and smoothed OBI momentum.
- Wavelet chop as a regime scaler for spread and taker thresholds.
- Cross-asset hedge logic tied to correlation with OSMIUM.

### 9.2 Drift Construction

`predicted_drift` is the core directional estimate.

It starts from trend and OBI, then gets adjusted by:

- trend bias from the rolling window
- orderflow signal from recent mid-price behavior
- an optional hedge ratio when OSMIUM exposure suggests systemic risk

This drift then determines target position and the fair-value anchor for quoting.

### 9.3 Execution Logic

The branch does three broad things:

1. Sets a target position using a tanh transform of predicted drift.
2. Builds passive bid/ask quotes around a reservation price.
3. Crosses the spread only when the drift exceeds a taker threshold.

The taker threshold is intentionally conservative unless the tape looks strong.

### 9.4 Regime Effects

Wavelet chop now influences both quoting and taking:

- In chop, the bot tends to widen and slow down.
- In cleaner directional conditions, it reacts sooner.

This is a small but important link between the professor-style regime observation and actual execution.

### 9.5 Aggressive OBI Scalping

There is also a smaller tactical branch that scalps mid-sized OBI on the Pepper book.

It is gated by:

- volatility conditions
- measurement variance
- Kalman gain
- timestamp
- imbalance size

### 9.6 Inventory-Aware Maker Replenishment

After the taker logic, Pepper still places passive inventory-aware maker orders.

Behavior:

- If the target position is above the current position, the bot leans toward a more aggressive bid.
- If the target position is below the current position, it steps back and fades the market.
- Order size is split into two layers so the bot does not send one blunt passive order.
- The bid/ask placement is adjusted by the current trend bias so the bot does not fight the dominant tape too hard.

This is the part that keeps Pepper continuously engaged even when it is not crossing the spread.

This helps when the order book is telling a short-lived directional story without needing the full target-position path.

## 10. Cross-Asset Risk Management

The bot uses OSMIUM and PEPPER together rather than independently.

Mechanisms:

- Correlation tracking between their returns.
- Correlation-based hedging when exposure becomes too aligned.
- Risk parity and drawdown scaling in the broader execution path.
- Volatility-based position scaling for both products.

The effect is that one leg can become a hedge or a risk amplifier depending on the current tape.

## 11. State Serialization

The bot serializes all state with JSON so it can survive the tick boundary.

This is necessary because the simulator calls the trader one tick at a time and otherwise loses all rolling history.

The serialization layer is intentionally defensive:

- It returns defaults if JSON is malformed.
- It rehydrates missing keys.
- It preserves expected container types.

At the end of each Pepper run, the bot also removes deprecated state keys from earlier experiments so the persisted payload does not keep drifting upward forever.

## 12. Efficiency Notes

The current version already includes a small efficiency pass:

- NumPy is used for repeated reductions and return calculations.
- pandas is used only for short bounded means.
- Repeated book scans were reduced where practical.

The strategy behavior stayed stable after those changes.

## 13. Results of the Current Model

Observed backtest results after the professor-style signal mapping and subsequent tuning:

- Rust round 2 total: `255920`
- Prosperity4btx total: `256156`

These are the current reference numbers for the deployed `trader.py` behavior.

## 14. File Naming Notes

The older experimental files were renamed to be descriptive:

- `john.py` was empty and is now `empty_placeholder.py`
- `Monk.py` contained the regression-based trader and is now `imbalance_regression_trader.py`

These names reflect function rather than personal labels.

## 15. Practical Takeaway

The current system is a hybrid of:

- directional Pepper trading
- mean-reverting OSMIUM market making
- wavelet regime detection
- Kalman adaptation
- cross-asset risk management
- lightweight execution smoothing

It is designed to keep the profitable structure while making the expensive decisions more selective.
