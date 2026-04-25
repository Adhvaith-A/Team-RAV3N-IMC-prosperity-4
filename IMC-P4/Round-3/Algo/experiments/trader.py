from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import math
import json
from typing import Any, Optional

import numpy as np

# TEAM RAV3N : Adhvaith A and Unnabh M
# Round 3 products: HYDROGEL_PACK, VELVETFRUIT_EXTRACT, VEV_* (options)
# Strategy: deviation-based mean reversion — buy when far below fair, sell when far above


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice, observation.transportFees,
                observation.exportTariff, observation.importTariff,
                observation.sugarPrice, observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ─── IMC Prosperity 4 tick constants ────────────────────────────────────────
TIMESTAMPS_PER_DAY = 1_000_000   # timestamps per day (ticks of 100ms each)
TICKS_PER_DAY = 10_000           # actual order book ticks per day (step=100)

# ─── Product constants ────────────────────────────────────────────────────────
HP   = "HYDROGEL_PACK"
VFE  = "VELVETFRUIT_EXTRACT"
VOUCHER_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000, "VEV_6500": 6500,
}

# ─── Position limits ──────────────────────────────────────────────────────────
POSITION_LIMITS = {
    HP: 200, VFE: 200,
    **{k: 300 for k in VOUCHER_STRIKES},
}

# ─── Round 3 timing calibration ───────────────────────────────────────────────
# TTE at start of round 3 = 5 trading days; decrements by 1 per round.
TTE_DAYS_START = 5.0

# ─── V17: Micro-Price + VPIN-lite parameters ────────────────────────────────
LIMIT_HYDROGEL       = 200    # AS inventory normaliser (matches HP_LIMIT)
AS_GAMMA             = 0.15   # AS risk aversion
MICRO_DECAY          = 0.5    # depth-weight decay for L1-L3
MICRO_SKEW_GAIN      = 0.7    # predictive skew on (micro - mid)
HP_BASE_SPREAD       = 3.0    # reservation-price ± base_spread for maker quotes
AS_VAR_WINDOW        = 50     # rolling window for variance(micro)
VPIN_WINDOW          = 50     # rolling toxicity bucket count
VPIN_TOXIC           = 0.70   # is_toxic threshold
VPIN_TOXIC_SPREAD_X  = 4.0    # widen spread when toxic


class Trader:

    # ─── tiny helpers ────────────────────────────────────────────────────────

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _ts_in_day(self, timestamp: int) -> int:
        """Return the tick position within the current day (0 – 999900)."""
        return timestamp % TIMESTAMPS_PER_DAY

    def _day_progress(self, timestamp: int) -> float:
        """Fraction of the current day elapsed: 0.0 (open) → 1.0 (close)."""
        return self._ts_in_day(timestamp) / TIMESTAMPS_PER_DAY

    def _restore_state(self, trader_data: str) -> dict:
        if not trader_data:
            return {}
        try:
            return json.loads(trader_data)
        except Exception:
            return {}

    def _serialize_state(self, data: dict) -> str:
        try:
            return json.dumps(data)
        except Exception:
            return "{}"

    # ─── Kalman filter for fair-value tracking ───────────────────────────────

    def _kalman_update(self, data: dict, symbol: str, measurement: float,
                       Q: float = 0.01, R: float = 4.0) -> float:
        """1-D Kalman update; returns updated state estimate."""
        key_x = f"kf_x_{symbol}"
        key_p = f"kf_p_{symbol}"
        x = data.get(key_x, measurement)
        p = data.get(key_p, 100.0)
        p_pred = p + Q
        k = p_pred / (p_pred + R)
        x_new = x + k * (measurement - x)
        p_new = (1 - k) * p_pred
        data[key_x] = x_new
        data[key_p] = p_new
        return x_new

    # ─── OBI helper ──────────────────────────────────────────────────────────

    def _obi(self, od: OrderDepth) -> float:
        bid_vol = sum(od.buy_orders.values())
        ask_vol = sum(abs(v) for v in od.sell_orders.values())
        total = max(1, bid_vol + ask_vol)
        return (bid_vol - ask_vol) / total

    # ─── V17: Micro-Price (Stoikov, depth-weighted L1-L3) ────────────────────

    def _micro_price(self, od: OrderDepth, decay: float = MICRO_DECAY) -> tuple:
        """Return (micro, delta_signal) where micro is depth-weighted across
        the top 3 levels and delta_signal = micro - naive_mid.

        Stoikov formula generalised: weight each side's price by the *opposite*
        side's quantity at that level, attenuated by exp(-decay*i) per level.
        """
        if not od.buy_orders or not od.sell_orders:
            return 0.0, 0.0
        bids = sorted(od.buy_orders.items(), key=lambda x: -x[0])[:3]
        asks = sorted(od.sell_orders.items(), key=lambda x:  x[0])[:3]
        levels = min(len(bids), len(asks))
        if levels == 0:
            return 0.0, 0.0
        num = 0.0
        den = 0.0
        for i in range(levels):
            w = math.exp(-decay * i)
            bp, bq = bids[i][0], abs(bids[i][1])
            ap, aq = asks[i][0], abs(asks[i][1])
            num += w * (aq * bp + bq * ap)
            den += w * (aq + bq)
        mid = 0.5 * (bids[0][0] + asks[0][0])
        if den <= 0:
            return mid, 0.0
        micro = num / den
        return micro, micro - mid

    # ─── V17: AS rolling variance using micro-price ──────────────────────────

    def _update_as_variance(self, data: dict, symbol: str, micro: float,
                            window: int = AS_VAR_WINDOW) -> float:
        """Rolling variance of micro-price over `window` ticks. Used in the
        AS reservation: r = micro - pos*gamma*var/Q."""
        key = f"as_micro_{symbol}"
        buf = data.get(key, [])
        buf.append(float(micro))
        if len(buf) > window:
            buf = buf[-window:]
        data[key] = buf
        if len(buf) < 5:
            return 1.0
        arr = np.asarray(buf, dtype=np.float64)
        return float(arr.var())

    # ─── V17: VPIN-lite (50-tick toxicity proxy) ─────────────────────────────

    def _update_vpin(self, data: dict, market_trades_by_sym: dict,
                     mid_by_sym: dict, window: int = VPIN_WINDOW) -> float:
        """Per-tick: classify each market trade as buy- or sell-initiated by
        comparing fill price to current mid; record per-tick imbalance
        |B-S|/(B+S); rolling-mean over `window` ticks = VPIN proxy."""
        vb = 0
        vs = 0
        for sym, trades in market_trades_by_sym.items():
            m = mid_by_sym.get(sym)
            if m is None or not trades:
                continue
            for t in trades:
                q = abs(t.quantity)
                if t.price >= m:
                    vb += q
                else:
                    vs += q
        tot = vb + vs
        imb = abs(vb - vs) / tot if tot > 0 else 0.0
        buf = data.get("vpin_buckets", [])
        buf.append(imb)
        if len(buf) > window:
            buf = buf[-window:]
        data["vpin_buckets"] = buf
        return (sum(buf) / len(buf)) if buf else 0.0

    # ─── Rolling deviation tracker ────────────────────────────────────────────

    def _update_price_window(self, data: dict, symbol: str, price: float,
                              window: int = 50) -> list:
        key = f"price_win_{symbol}"
        buf = data.get(key, [])
        buf.append(price)
        if len(buf) > window:
            buf = buf[-window:]
        data[key] = buf
        return buf

    # ─── Sizing helper ────────────────────────────────────────────────────────

    def _aggressive_qty(self, deviation: float, threshold: float,
                        cap: int, scale: float = 1.0) -> int:
        """Map deviation depth beyond threshold to a buy/sell quantity."""
        excess = abs(deviation) - threshold
        frac = self._clamp(excess / (threshold * 2.0), 0.0, 1.0)
        return max(1, int(cap * frac * scale))

    # ─────────────────────────────────────────────────────────────────────────
    # HYDROGEL_PACK
    # Fair value is anchored at 10 000 with small mean-reverting oscillations
    # (±100 max observed). Spread ≈ 16. Position limit 200.
    #
    # Buy-low / sell-high thresholds derived from CSV analysis:
    #   deviation < -50  → aggressive take (52 % forward win rate)
    #   deviation < -20  → passive maker shift toward bid
    #   deviation >  50  → aggressive take on bid side
    #   deviation >  20  → passive maker shift toward ask
    # ─────────────────────────────────────────────────────────────────────────

    HP_FAIR        = 10_000
    HP_LIMIT       = 200
    HP_AGG_BUY     = -50   # 60% win rate at 20-tick horizon (CSV calibrated)
    HP_AGG_BUY_MAX = -70   # ultra-aggressive zone (>59% at 20t, only 925 obs)
    HP_AGG_SELL    =  50   # 56% win at 20t; at >=70 it's 82.8% at 5t (CSV)
    HP_AGG_SELL_MAX = 70   # ultra-sell zone — slam bids hard
    HP_PASS_BUY    = -20
    HP_PASS_SELL   =  20

    def _hydrogel(self, od: OrderDepth, pos: int, timestamp: int, data: dict) -> list[Order]:
        orders: list[Order] = []
        if not od.buy_orders or not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        limit      = self.HP_LIMIT
        buy_cap    = max(0, limit - pos)
        sell_cap   = max(0, limit + pos)

        # Kalman-filtered fair value (anchored near 10000)
        fair = self._kalman_update(data, HP, mid, Q=0.5, R=16.0)
        # Hard-anchor: Hydrogel has a strong attractor at 10000
        fair = 0.85 * self.HP_FAIR + 0.15 * fair
        deviation = mid - fair

        # ── Timing gate ──────────────────────────────────────────────────────
        dp = self._day_progress(timestamp)

        # Warm-up: first 3 % of day — skip aggressive trades
        is_warmup    = dp < 0.03

        # Scale aggressiveness by day progress (ramp up after warm-up)
        if is_warmup:
            aggr_scale = 0.2
        else:
            aggr_scale = 1.0

        # ── 1. Aggressive taker: price is deeply mis-priced ──────────────────
        # Ultra zone (dev <= -70): CSV shows 59%+ win at 20t — take everything available
        # Normal zone (dev <= -50): CSV shows 60% win at 20t — proportional sizing
        if deviation <= self.HP_AGG_BUY and buy_cap > 0 and not is_warmup:
            ultra = deviation <= self.HP_AGG_BUY_MAX
            qty = buy_cap if ultra else self._aggressive_qty(deviation, abs(self.HP_AGG_BUY), buy_cap, aggr_scale)
            asks_asc = sorted(od.sell_orders.keys())
            for ask in asks_asc:
                price_ok = ask <= fair + 2 if not ultra else ask <= best_ask + 1
                if price_ok and qty > 0:
                    fill = min(abs(od.sell_orders[ask]), qty)
                    if fill > 0:
                        orders.append(Order(HP, ask, fill))
                        qty -= fill
                        buy_cap -= fill

        # Ultra-sell zone (dev >= 70): CSV shows 82.8% win at 5t — slam bids hard
        # Normal sell zone (dev >= 50): CSV shows 56% win at 20t
        if deviation >= self.HP_AGG_SELL and sell_cap > 0 and not is_warmup:
            ultra = deviation >= self.HP_AGG_SELL_MAX
            qty = sell_cap if ultra else self._aggressive_qty(deviation, self.HP_AGG_SELL, sell_cap, aggr_scale)
            bids_desc = sorted(od.buy_orders.keys(), reverse=True)
            for bid in bids_desc:
                price_ok = bid >= fair - 2 if not ultra else bid >= best_bid - 1
                if price_ok and qty > 0:
                    fill = min(od.buy_orders[bid], qty)
                    if fill > 0:
                        orders.append(Order(HP, bid, -fill))
                        qty -= fill
                        sell_cap -= fill

        # ── 2. V18 Dynamic Maker: Micro-Price + Dynamic Gamma AS + Asymmetric Spreads
        # Stoikov micro across L1-L3 → predictive leading signal.
        micro, delta_signal = self._micro_price(od, decay=MICRO_DECAY)
        if micro <= 0.0:                        # degenerate book → fall back
            micro = mid

        # Rolling variance of micro-price (replaces mid-based variance, V17 spec).
        as_var = self._update_as_variance(data, HP, micro)

        # V18: Dynamic Gamma - increases exponentially as inventory approaches limit
        # Gamma stays low when inventory is safe, spikes aggressively to dump toxic inventory
        dynamic_gamma = 0.05 + 0.20 * ((abs(pos) / float(LIMIT_HYDROGEL)) ** 2)

        # AS reservation: r = micro − pos·γ·σ²/Q   (V18 dynamic inventory penalty)
        reservation_price = micro - (pos * dynamic_gamma * as_var / float(LIMIT_HYDROGEL))
        # Predictive skew: front-run mid drift implied by book imbalance
        reservation_price += MICRO_SKEW_GAIN * delta_signal
        # Residual deviation lean (preserve V16 dev-zone bias on top of AS)
        if deviation <= self.HP_PASS_BUY:
            reservation_price += 1.0
        elif deviation >= self.HP_PASS_SELL:
            reservation_price -= 1.0

        is_toxic = data.get("is_toxic", False)
        base_spread = HP_BASE_SPREAD
        if is_toxic:
            # Toxic: widen 4×, drop quotes to bb/ba — pure liquidity backstop.
            spread = base_spread * VPIN_TOXIC_SPREAD_X
            my_bid = min(int(round(reservation_price - spread)), best_bid)
            my_ask = max(int(round(reservation_price + spread)), best_ask)
        else:
            # V18: Asymmetric Spreads based on delta_signal momentum
            # If delta_signal > 0 (upward momentum): tighten Bid (buy faster), widen Ask
            # If delta_signal < 0 (downward momentum): tighten Ask (sell faster), widen Bid
            bid_skew = -delta_signal * 0.5
            ask_skew = delta_signal * 0.5

            my_bid = int(math.floor(reservation_price - (base_spread / 2.0) + bid_skew))
            my_ask = int(math.ceil(reservation_price + (base_spread / 2.0) + ask_skew))

            # Clamp inside the market spread
            my_bid = min(my_bid, best_ask - 1)
            my_ask = max(my_ask, best_bid + 1)
            if best_bid + 1 < best_ask:
                my_bid = max(my_bid, best_bid + 1)
            if best_ask - 1 > best_bid:
                my_ask = min(my_ask, best_ask - 1)

        if my_bid >= my_ask:
            my_bid = my_ask - 1

        if buy_cap > 0:
            layer1 = max(1, int(buy_cap * 0.6))
            layer2 = buy_cap - layer1
            orders.append(Order(HP, my_bid, layer1))
            if layer2 > 0 and my_bid - 1 > 0:
                orders.append(Order(HP, my_bid - 1, layer2))

        if sell_cap > 0:
            layer1 = max(1, int(sell_cap * 0.6))
            layer2 = sell_cap - layer1
            orders.append(Order(HP, my_ask, -layer1))
            if layer2 > 0:
                orders.append(Order(HP, my_ask + 1, -layer2))

        return orders

    # ─────────────────────────────────────────────────────────────────────────
    # VELVETFRUIT_EXTRACT
    # Fair value drifts upward (~5250 → 5296 over 3 days, ~+15/day).
    # Spread ≈ 5. Position limit 200.
    #
    # Thresholds from CSV analysis:
    #   deviation < -40  → aggressive buy (58 % win rate)
    #   deviation < -15  → passive buy lean
    #   deviation >  30  → passive sell lean (>40 may keep trending — be careful)
    # ─────────────────────────────────────────────────────────────────────────

    VFE_FAIR_SEED  = 5250
    VFE_LIMIT      = 200
    # CSV analysis: dev zone ±15 = 0.5% of ticks; ±10 = 5.3% of ticks; ±5 = 44%
    # dev<=-15: 57.8% win at 10t (best reliable signal)
    # dev<=-10: 49.8% win (weak, skip aggressive take)
    VFE_AGG_BUY    = -12   # hits ~1% of ticks; take aggressively
    VFE_AGG_SELL   =  12   # symmetric
    VFE_PASS_BUY   =  -5   # lean bids when below this
    VFE_PASS_SELL  =   5   # lean asks when above this

    def _velvetfruit(self, od: OrderDepth, pos: int, timestamp: int, data: dict) -> list[Order]:
        orders: list[Order] = []
        if not od.buy_orders or not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0

        limit   = self.VFE_LIMIT
        buy_cap = max(0, limit - pos)
        sell_cap = max(0, limit + pos)

        # Track a slower Kalman estimate of fair value (captures the upward drift)
        # Q=2 allows gradual drift; R=16 smooths noise
        fair = self._kalman_update(data, VFE, mid, Q=2.0, R=16.0)

        # Also track a short-horizon EMA to capture recent momentum
        ema_key = f"ema_{VFE}"
        prev_ema = data.get(ema_key, mid)
        ema = 0.15 * mid + 0.85 * prev_ema
        data[ema_key] = ema

        # Momentum signal: ema above/below fair → short-term trend
        momentum = ema - fair

        deviation = mid - fair
        obi = self._obi(od)

        # ── Timing gate ──────────────────────────────────────────────────────
        dp = self._day_progress(timestamp)
        is_warmup = dp < 0.03

        if is_warmup:
            aggr_scale = 0.25
        else:
            aggr_scale = 1.0

        # ── 1. Aggressive taker ────────────────────────────────────────────────
        if deviation <= self.VFE_AGG_BUY and buy_cap > 0 and not is_warmup:
            # Deep dip confirmed → buy at ask (58 %+ win rate from CSV data)
            qty = self._aggressive_qty(deviation, abs(self.VFE_AGG_BUY), buy_cap, aggr_scale)
            asks_asc = sorted(od.sell_orders.keys())
            for ask in asks_asc:
                if ask <= fair + 3 and qty > 0:
                    fill = min(abs(od.sell_orders[ask]), qty)
                    if fill > 0:
                        orders.append(Order(VFE, ask, fill))
                        qty -= fill
                        buy_cap -= fill

        if deviation >= self.VFE_AGG_SELL and sell_cap > 0 and not is_warmup:
            # Data shows >40 can keep trending up so only sell at 30–40 zone
            if deviation < 45 or momentum < -5:
                qty = self._aggressive_qty(deviation, self.VFE_AGG_SELL, sell_cap, aggr_scale)
                bids_desc = sorted(od.buy_orders.keys(), reverse=True)
                for bid in bids_desc:
                    if bid >= fair - 3 and qty > 0:
                        fill = min(od.buy_orders[bid], qty)
                        if fill > 0:
                            orders.append(Order(VFE, bid, -fill))
                            qty -= fill
                            sell_cap -= fill

        # ── 2. Passive maker ──────────────────────────────────────────────────
        inv_skew = (pos / float(limit)) * 3.0

        # Momentum bias: if fast EMA above fair, lean ask; if below, lean bid
        trend_offset = self._clamp(momentum * 0.5, -3.0, 3.0)

        if deviation <= self.VFE_PASS_BUY:
            my_bid = min(int(fair - 1 - inv_skew + trend_offset), best_ask - 1)
            my_ask = max(int(fair + 3 - inv_skew + trend_offset), best_bid + 1)
        elif deviation >= self.VFE_PASS_SELL:
            my_bid = min(int(fair - 3 - inv_skew + trend_offset), best_ask - 1)
            my_ask = max(int(fair + 1 - inv_skew + trend_offset), best_bid + 1)
        else:
            my_bid = min(int(fair - 2 - inv_skew + trend_offset), best_ask - 1)
            my_ask = max(int(fair + 2 - inv_skew + trend_offset), best_bid + 1)

        if my_bid >= my_ask:
            my_bid = my_ask - 1

        if buy_cap > 0:
            layer1 = max(1, int(buy_cap * 0.65))
            layer2 = buy_cap - layer1
            orders.append(Order(VFE, my_bid, layer1))
            if layer2 > 0 and my_bid - 1 > 0:
                orders.append(Order(VFE, my_bid - 1, layer2))

        if sell_cap > 0:
            layer1 = max(1, int(sell_cap * 0.65))
            layer2 = sell_cap - layer1
            orders.append(Order(VFE, my_ask, -layer1))
            if layer2 > 0:
                orders.append(Order(VFE, my_ask + 1, -layer2))

        return orders

    # ─────────────────────────────────────────────────────────────────────────
    # VEV_XXXX vouchers (call options on VELVETFRUIT_EXTRACT)
    # Simplified Black-Scholes market making:
    #   TTE decreases from 5 days at round start → 0 at expiry.
    #   We compute BS fair value, then:
    #     - buy vouchers trading below BS fair - edge
    #     - sell vouchers trading above BS fair + edge
    # ─────────────────────────────────────────────────────────────────────────

    VEV_LIMIT = 300
    INTERNAL_VEV_CAP = 40  # V17.2: Tightened from 60 to prevent toxic bags during vol shocks
    # IV calibrated from CSV: day0=25.2%, day1=27.1%, day2=28.8% (ATM VEV_5200)
    VEV_VOL_BY_DAY = [0.252, 0.271, 0.288]
    # V17.2: IV thresholds for ATM and OTM options (loosened to avoid hyper-trading noise)
    iv_threshold_atm = 0.020  # V17.2: Loosened from 0.015
    iv_threshold_otm = 0.030  # V17.2: Loosened from 0.025

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _bs_call(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0:
            return max(0.0, S - K)
        if S <= 0 or sigma <= 0:
            return max(0.0, S - K)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self._norm_cdf(d1) - K * self._norm_cdf(d2)

    def _tte(self, timestamp: int, data: dict) -> float:
        """TTE in years. Tracks day via state since timestamps restart each day."""
        day = data.get("current_day", 0)
        remaining_days = max(0.0, TTE_DAYS_START - day - timestamp / TIMESTAMPS_PER_DAY)
        return remaining_days / 252.0

    def _update_day(self, timestamp: int, data: dict) -> int:
        """Detect day rollover (timestamp resets to ~0) and increment day counter."""
        prev_ts = data.get("prev_timestamp", -1)
        if prev_ts > 0 and timestamp < prev_ts - 500_000:
            data["current_day"] = data.get("current_day", 0) + 1
        data["prev_timestamp"] = timestamp
        return data.get("current_day", 0)

    def _voucher(self, symbol: str, strike: int, od: OrderDepth,
                 pos: int, timestamp: int, data: dict, S: float) -> list[Order]:
        orders: list[Order] = []
        if not od.buy_orders or not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0

        day = data.get("current_day", 0)
        sigma = self.VEV_VOL_BY_DAY[min(day, 2)]
        T = self._tte(timestamp, data)
        fair = self._bs_call(S, strike, T, sigma)

        limit    = self.VEV_LIMIT
        internal_cap = self.INTERNAL_VEV_CAP  # V17.2: Tighter internal cap for risk mitigation
        buy_cap  = max(0, limit - pos)
        sell_cap = max(0, limit + pos)

        # Skip deep OTM where flat-vol underprices (vol smile correction needed)
        if fair < 1.0 or strike >= 5500:
            return orders

        # V17.2: IV threshold based on moneyness (ATM vs OTM)
        # ATM strikes: 5000-5400, OTM: 5500+
        is_atm = 5000 <= strike <= 5400
        iv_threshold = self.iv_threshold_atm if is_atm else self.iv_threshold_otm

        mkt_spread = best_ask - best_bid
        # Edge: at least half market spread to avoid adverse selection
        edge = max(1.5, mkt_spread * 0.6)
        # Wider near expiry (gamma spikes)
        if T < 1.0 / 252:
            edge = max(edge, 4.0)

        # Aggressive arb: only cross spread when misprice > edge + IV threshold
        # V17.2: Apply IV threshold buffer and internal position cap
        effective_edge_buy = edge + iv_threshold * fair
        effective_edge_sell = edge + iv_threshold * fair

        # V17.2: Clamp buy/sell caps to internal cap to prevent toxic bag accumulation
        buy_cap_clamped = min(buy_cap, max(0, internal_cap - pos))
        sell_cap_clamped = min(sell_cap, max(0, internal_cap + pos))

        if best_ask <= fair - effective_edge_buy and buy_cap_clamped > 0:
            qty = min(buy_cap_clamped, abs(od.sell_orders[best_ask]))
            if qty > 0:
                orders.append(Order(symbol, best_ask, qty))

        if best_bid >= fair + effective_edge_sell and sell_cap_clamped > 0:
            qty = min(sell_cap_clamped, od.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order(symbol, best_bid, -qty))

        # Passive quotes: quote inside the spread to be at top of the queue
        # For wide-spread products (deep ITM), force at least 1 tick inside
        # so we're best bid/ask, not joining the back of the existing book
        q_bid = int(math.floor(fair - edge))
        q_ask = int(math.ceil(fair + edge))
        if mkt_spread >= 4:
            q_bid = max(q_bid, best_bid + 1)
            q_ask = min(q_ask, best_ask - 1)
        else:
            q_bid = max(q_bid, best_bid)
            q_ask = min(q_ask, best_ask)
        q_bid = min(q_bid, best_ask - 1)
        q_ask = max(q_ask, best_bid + 1)
        if q_bid >= q_ask:
            q_bid = q_ask - 1

        layer = max(1, min(buy_cap, 10))
        if buy_cap > 0:
            orders.append(Order(symbol, q_bid, layer))
        if sell_cap > 0:
            orders.append(Order(symbol, q_ask, -layer))

        return orders

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result: dict[Symbol, list[Order]] = {}
        data = self._restore_state(state.traderData)

        # Must call first — detects day rollover so TTE and IV stay correct
        self._update_day(state.timestamp, data)

        # Track VELVETFRUIT mid price for use as underlying in BS voucher pricing
        vfe_od = state.order_depths.get(VFE)
        if vfe_od and vfe_od.buy_orders and vfe_od.sell_orders:
            S = (max(vfe_od.buy_orders.keys()) + min(vfe_od.sell_orders.keys())) / 2.0
            data["vfe_S"] = S
        else:
            S = data.get("vfe_S", self.VFE_FAIR_SEED)

        # ─── V17: VPIN Shield ─────────────────────────────────────────────────
        # Classify all market trades each tick → buy/sell-initiated, roll 50.
        mid_by_sym: dict = {}
        for sym, od in state.order_depths.items():
            if od.buy_orders and od.sell_orders:
                mid_by_sym[sym] = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
        current_vpin = self._update_vpin(data, state.market_trades, mid_by_sym)
        data["current_vpin"] = current_vpin
        is_toxic = current_vpin > VPIN_TOXIC
        data["is_toxic"] = is_toxic

        for product, od in state.order_depths.items():
            pos = state.position.get(product, 0)

            if product == HP:
                result[product] = self._hydrogel(od, pos, state.timestamp, data)

            elif product == VFE:
                result[product] = self._velvetfruit(od, pos, state.timestamp, data)

            elif product in VOUCHER_STRIKES:
                strike = VOUCHER_STRIKES[product]
                result[product] = self._voucher(product, strike, od, pos, state.timestamp, data, S)

        trader_data = self._serialize_state(data)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
