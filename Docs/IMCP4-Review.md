# IMCP4 Code Review And Rewrite Decision

This review covers current code plus historical/winner references to decide between incremental refactor and full rewrite.

## High-Impact Findings

1. Data model bug in conversion observations.
- In [datamodel.py](datamodel.py#L24), constructor parameters are `sunlight` and `humidity`.
- In [datamodel.py](datamodel.py#L30), [datamodel.py](datamodel.py#L31), code assigns `sugarPrice` and `sunlightIndex`, which are undefined.
- This can break observation serialization in [trader.py](trader.py#L92).

2. Current trader is strong but tightly coupled to one round's products and timing.
- Product constants and assumptions are hardcoded near [trader.py](trader.py#L467).
- Time-gated logic appears throughout execution paths (for example [trader.py](trader.py#L697), [trader.py](trader.py#L705), [trader.py](trader.py#L1089)).
- Result: high performance in known regime, weaker portability.

3. Feature interaction complexity is high.
- Dynamic kalman, risk parity, wavelet, correlation hedge, and aggressive OBI are all active pathways.
- Configuration begins around [trader.py](trader.py#L486), but interactions are difficult to ablate cleanly.

4. Some winner/reference files are useful conceptually but unsafe to port blindly.
- [PreviousWinner/FrankfurtHedgehogs_polished.py](PreviousWinner/FrankfurtHedgehogs_polished.py) contains correctness issues and broad try/except style patterns.
- [PreviousWinner/round2-trader-stable.py](PreviousWinner/round2-trader-stable.py) has clean basket-hedge structure and is good as a structural template.

## Decision Matrix

### Proposal A: Incremental refactor of `trader.py`

Pros:
- Lowest implementation risk.
- Preserves known PnL behavior.
- Faster to validate under time pressure.

Cons:
- Existing coupling remains unless refactor is disciplined.
- Harder long-term maintenance than a clean architecture.

Recommended scope:
1. Split into modules within same file first:
- state, features, risk, execution, product routers.
2. Convert hardcoded constants to product config maps.
3. Add strict feature flags + ablation harness.
4. Fix `datamodel.py` bug immediately.

### Proposal B: Full rewrite

Pros:
- Clean architecture from day one.
- Better portability across rounds/products.
- Easier testing and extension.

Cons:
- Highest short-term regression risk.
- Larger validation burden before deployment.

Recommended architecture:
1. `ProductAdapter` (symbols, limits, schedule)
2. `SignalEngine` (fair value, OBI, trend, regime)
3. `RiskEngine` (limits, drawdown scaling, correlation caps)
4. `ExecutionEngine` (taker, maker, inventory skew)
5. `StateStore` (typed state schema with versioning)

## Recommendation

Use a two-track path:
1. Primary for immediate competition safety: Proposal A.
2. Secondary in parallel for medium-term edge: Proposal B prototype in separate file.

This gives fast iteration without risking the current validated production path.

## Extra Evidence From IMCP3 Rebuild

The isolated IMCP3 round-2 rebuild reinforced a few architecture lessons:

1. Structural products want structural logic.
- A clean basket-target engine beat the over-engineered IMCP3 baseline.
- Translation: when a product has a hard synthetic relationship, start with direct spread pricing and simple hedge propagation before layering adaptive filters.

2. Product routers matter.
- The IMCP3 trader improved once round-1 market making and round-2 basket logic were separated cleanly.
- The same split would make `trader.py` easier to tune and safer to ablate.

3. A rewrite should preserve per-product simplicity.
- The current IMCP4 file is strong but densely coupled.
- A rewrite should keep one orchestrator but move each product into its own signal, target, and execution path.

## Immediate Action Items

1. `Completed:` fix [datamodel.py](datamodel.py#L24) constructor fields to match the intended schema.
2. Create shared product config map and route current logic through it.
3. Build one-command ablation script for feature toggles.
4. Move time-window constants into config.
5. Keep full rewrite prototype in a separate file until parity is demonstrated.
