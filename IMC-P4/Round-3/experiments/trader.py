"""
IMC Prosperity – Round 3 Trader
Products: HYDROGEL_PACK · VELVETFRUIT_EXTRACT · VEV_XXXX (10 call-option vouchers)

Strategy overview
─────────────────
HYDROGEL_PACK   : Mean-reversion market maker + momentum overlay
VELVETFRUIT_EXTRACT: Market maker around VWAP fair value + liquidity taking
VOUCHERS        : Black-Scholes pricing → IV surface fit → arbitrage + MM
                  with an optional delta-hedge leg in VEV
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VOUCHER_STRIKES: Dict[str, int] = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}

LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{k: 300 for k in VOUCHER_STRIKES},
}

# ─── Time-to-expiry calibration ───────────────────────────────────────────────
# Round 1 → TTE=7d, Round 2 → TTE=6d, Round 3 → TTE=5d
# Each simulation round spans ~1_000_000 timestamps.
TTE_START_DAYS: float = 5.0        # change to 6 for R2, 7 for R1
TIMESTAMPS_PER_DAY: int = 1_000_000

# ─── Tuneable hyperparameters ─────────────────────────────────────────────────
HYDROGEL_SPREAD        = 2          # half-spread for HG market-making
HYDROGEL_TAKE_EDGE     = 1          # take liquidity if price beats FV by this
HYDROGEL_MM_QTY        = 30         # passive order size per side
HYDROGEL_TAKE_QTY      = 50         # aggressive fill size

VEV_SPREAD             = 3
VEV_TAKE_EDGE          = 2
VEV_MM_QTY             = 30
VEV_TAKE_QTY           = 50

# Voucher thresholds
VOL_SMOOTHING_WINDOW   = 15         # timesteps to smooth IV estimate
VOUCHER_EDGE_PCT       = 0.008      # minimum edge (as % of fair) to take
VOUCHER_EDGE_MIN       = 1.5        # absolute floor on the edge
VOUCHER_TAKE_QTY       = 60         # max aggressive fill per voucher
VOUCHER_MM_QTY         = 15         # passive order size
VOUCHER_MM_SPREAD_PCT  = 0.006      # half-spread as fraction of fair

# Delta-hedge VEV against voucher book (True = hedge net delta)
DELTA_HEDGE            = True
DELTA_HEDGE_BAND       = 10         # don't hedge when net delta < this


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    """Standard-normal CDF using math.erf (no numpy required)."""
    return 0.5 * (1.0 + math.erf(x * 0.7071067811865476))  # 1/√2


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """European call price (Black-Scholes)."""
    if T <= 0.0:
        return max(S - K, 0.0)
    if sigma < 1e-9:
        return max(S - K * math.exp(-r * T), 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Call delta = N(d1)."""
    if T <= 0.0:
        return 1.0 if S > K else 0.0
    if sigma < 1e-9:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _ncdf(d1)


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    """Call vega (∂C/∂σ). Always ≥ 0."""
    if T <= 0.0 or sigma < 1e-9:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
) -> Optional[float]:
    """
    Newton-Raphson implied volatility.
    Returns None if the problem is degenerate or diverges.
    """
    if T <= 0.0:
        return None
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    # Price below intrinsic → no real IV
    if market_price < intrinsic - 0.5:
        return None
    # Clamp to slightly above intrinsic to keep log valid
    market_price = max(market_price, intrinsic + 1e-6)

    sigma = 0.40  # warm start
    for _ in range(60):
        price = bs_call(S, K, T, sigma, r)
        vega  = bs_vega(S, K, T, sigma)
        if vega < 1e-8:
            break
        diff = price - market_price
        sigma -= diff / vega
        sigma = max(0.005, min(sigma, 6.0))
        if abs(diff) < 0.05:
            return sigma
    return sigma if 0.005 < sigma < 6.0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def mid_price(od: OrderDepth) -> Optional[float]:
    if not od.buy_orders or not od.sell_orders:
        return None
    return (max(od.buy_orders) + min(od.sell_orders)) / 2.0


def best_bid_ask(od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    bb = max(od.buy_orders)  if od.buy_orders  else None
    ba = min(od.sell_orders) if od.sell_orders else None
    return bb, ba


def vwap(od: OrderDepth) -> Optional[float]:
    """Volume-weighted mid price from entire visible book."""
    num = den = 0.0
    for price, qty in od.buy_orders.items():
        num += price * qty; den += qty
    for price, qty in od.sell_orders.items():
        num += price * abs(qty); den += abs(qty)
    return num / den if den > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Trader
# ─────────────────────────────────────────────────────────────────────────────

class Trader:
    """
    State is carried across ticks via the JSON traderData string.
    All mutable lists are restored at the start of every run() call.
    """

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _restore(self, raw: str) -> None:
        try:
            d = json.loads(raw) if raw else {}
        except Exception:
            d = {}
        self.hg_px:   List[float] = d.get("hg_px",  [])
        self.vev_px:  List[float] = d.get("vev_px", [])
        self.iv_hist: List[float] = d.get("iv_hist", [])

    def _save(self) -> str:
        return json.dumps({
            "hg_px":  self.hg_px[-MAX_HIST:],
            "vev_px": self.vev_px[-MAX_HIST:],
            "iv_hist": self.iv_hist[-30:],
        })

    @staticmethod
    def _fair(prices: List[float], window: int = 20) -> float:
        tail = prices[-window:] if len(prices) >= window else prices
        return sum(tail) / len(tail)

    # ── HYDROGEL_PACK ─────────────────────────────────────────────────────────

    def _trade_hydrogel(self, state: TradingState) -> List[Order]:
        prod = "HYDROGEL_PACK"
        od   = state.order_depths.get(prod)
        if od is None:
            return []

        mp = mid_price(od) or vwap(od)
        if mp is None:
            return []

        self.hg_px.append(mp)
        fv  = self._fair(self.hg_px, 20)
        pos = state.position.get(prod, 0)
        lim = LIMITS[prod]
        bb, ba = best_bid_ask(od)
        orders: List[Order] = []

        # Aggressive takes ─ only when market clearly wrong vs FV
        if ba is not None and ba < fv - HYDROGEL_TAKE_EDGE:
            qty = min(abs(od.sell_orders[ba]), lim - pos, HYDROGEL_TAKE_QTY)
            if qty > 0:
                orders.append(Order(prod, ba, qty))
                pos += qty

        if bb is not None and bb > fv + HYDROGEL_TAKE_EDGE:
            qty = min(od.buy_orders[bb], lim + pos, HYDROGEL_TAKE_QTY)
            if qty > 0:
                orders.append(Order(prod, bb, -qty))
                pos -= qty

        # Passive market-making
        buy_qty  = min(HYDROGEL_MM_QTY, lim - pos)
        sell_qty = min(HYDROGEL_MM_QTY, lim + pos)
        if buy_qty  > 0:
            orders.append(Order(prod, math.floor(fv)  - HYDROGEL_SPREAD,  buy_qty))
        if sell_qty > 0:
            orders.append(Order(prod, math.ceil(fv)   + HYDROGEL_SPREAD, -sell_qty))

        return orders

    # ── VELVETFRUIT_EXTRACT ───────────────────────────────────────────────────

    def _trade_vev(self, state: TradingState) -> Tuple[List[Order], Optional[float]]:
        prod = "VELVETFRUIT_EXTRACT"
        od   = state.order_depths.get(prod)
        if od is None:
            return [], None

        mp = mid_price(od) or vwap(od)
        if mp is None:
            return [], None

        self.vev_px.append(mp)
        fv  = self._fair(self.vev_px, 20)
        pos = state.position.get(prod, 0)
        lim = LIMITS[prod]
        bb, ba = best_bid_ask(od)
        orders: List[Order] = []

        if ba is not None and ba < fv - VEV_TAKE_EDGE:
            qty = min(abs(od.sell_orders[ba]), lim - pos, VEV_TAKE_QTY)
            if qty > 0:
                orders.append(Order(prod, ba, qty))
                pos += qty

        if bb is not None and bb > fv + VEV_TAKE_EDGE:
            qty = min(od.buy_orders[bb], lim + pos, VEV_TAKE_QTY)
            if qty > 0:
                orders.append(Order(prod, bb, -qty))
                pos -= qty

        buy_qty  = min(VEV_MM_QTY, lim - pos)
        sell_qty = min(VEV_MM_QTY, lim + pos)
        if buy_qty  > 0:
            orders.append(Order(prod, math.floor(fv) - VEV_SPREAD,  buy_qty))
        if sell_qty > 0:
            orders.append(Order(prod, math.ceil(fv)  + VEV_SPREAD, -sell_qty))

        return orders, fv

    # ── IV surface estimation ─────────────────────────────────────────────────

    def _estimate_iv(
        self,
        state: TradingState,
        S: float,
        T: float,
    ) -> float:
        """
        Weighted-average implied vol across all vouchers that have a market.
        Near-ATM options carry more weight (lower moneyness → higher weight).
        Falls back to a rolling average of past estimates, then to 0.35.
        """
        iv_samples: List[Tuple[float, float]] = []  # (iv, weight)

        for name, K in VOUCHER_STRIKES.items():
            od = state.order_depths.get(name)
            if od is None:
                continue
            mp = mid_price(od)
            if mp is None:
                continue
            iv = implied_vol(mp, S, K, T)
            if iv is None:
                continue
            moneyness = abs(math.log(S / K))          # |ln(F/K)|
            weight    = math.exp(-4.0 * moneyness)    # ATM → weight≈1, OTM → lower
            iv_samples.append((iv, weight))

        if iv_samples:
            tw = sum(w for _, w in iv_samples)
            sigma = sum(iv * w for iv, w in iv_samples) / tw
            # Sanity-clamp
            sigma = max(0.05, min(sigma, 3.0))
            self.iv_hist.append(sigma)

        if not self.iv_hist:
            return 0.35                                # cold start

        # Exponentially-smoothed (simple window average here for clarity)
        window = min(len(self.iv_hist), VOL_SMOOTHING_WINDOW)
        return sum(self.iv_hist[-window:]) / window

    # ── Voucher trading ───────────────────────────────────────────────────────

    def _trade_vouchers(
        self,
        state: TradingState,
        S: float,
        T: float,
    ) -> Dict[str, List[Order]]:
        """
        For each voucher:
          1. Compute BS fair value using smoothed IV.
          2. Aggressively take if market bid > fair+edge (sell) or ask < fair-edge (buy).
          3. Post passive MM orders around fair value.
        Returns a dict keyed by product name.
        """
        sigma = self._estimate_iv(state, S, T)
        result: Dict[str, List[Order]] = {}

        for name, K in VOUCHER_STRIKES.items():
            od = state.order_depths.get(name)
            if od is None:
                result[name] = []
                continue

            pos = state.position.get(name, 0)
            lim = LIMITS[name]
            bb, ba = best_bid_ask(od)
            fair    = bs_call(S, K, T, sigma)
            edge    = max(VOUCHER_EDGE_MIN, fair * VOUCHER_EDGE_PCT)
            mm_sprd = max(2, round(fair * VOUCHER_MM_SPREAD_PCT))
            orders: List[Order] = []

            # ── Aggressive buys (ask is too cheap) ────────────────────────────
            if ba is not None and ba < fair - edge:
                qty = min(abs(od.sell_orders[ba]), lim - pos, VOUCHER_TAKE_QTY)
                if qty > 0:
                    orders.append(Order(name, ba, qty))
                    pos += qty

            # ── Aggressive sells (bid is too rich) ────────────────────────────
            if bb is not None and bb > fair + edge:
                qty = min(od.buy_orders[bb], lim + pos, VOUCHER_TAKE_QTY)
                if qty > 0:
                    orders.append(Order(name, bb, -qty))
                    pos -= qty

            # ── Passive market-making ─────────────────────────────────────────
            buy_qty  = min(VOUCHER_MM_QTY, lim - pos)
            sell_qty = min(VOUCHER_MM_QTY, lim + pos)
            fair_int = round(fair)
            if buy_qty  > 0 and fair_int - mm_sprd > 0:
                orders.append(Order(name, fair_int - mm_sprd,  buy_qty))
            if sell_qty > 0:
                orders.append(Order(name, fair_int + mm_sprd, -sell_qty))

            result[name] = orders

        return result

    # ── Delta hedge VEV ───────────────────────────────────────────────────────

    def _delta_hedge_orders(
        self,
        state: TradingState,
        S: float,
        T: float,
        sigma: float,
        existing_vev_orders: List[Order],
    ) -> List[Order]:
        """
        Compute net delta of the entire voucher book (positions + pending orders),
        then issue a VEV order to neutralise it within position limits.
        """
        net_delta = 0.0
        for name, K in VOUCHER_STRIKES.items():
            pos = state.position.get(name, 0)
            δ   = bs_delta(S, K, T, sigma)
            net_delta += pos * δ

        # Target: net_delta + hedge_qty * 1.0 ≈ 0
        # So hedge_qty ≈ -net_delta
        target_hedge = -round(net_delta)
        if abs(target_hedge) < DELTA_HEDGE_BAND:
            return []

        prod    = "VELVETFRUIT_EXTRACT"
        cur_pos = state.position.get(prod, 0)
        lim     = LIMITS[prod]

        # Account for orders we're already sending this tick
        pending = sum(o.quantity for o in existing_vev_orders)
        cur_pos_effective = cur_pos + pending

        # Clamp to position limit
        new_target = max(-lim, min(lim, cur_pos_effective + target_hedge))
        hedge_qty  = new_target - cur_pos_effective

        if hedge_qty == 0:
            return []

        od = state.order_depths.get(prod)
        if od is None:
            return []
        bb, ba = best_bid_ask(od)

        orders: List[Order] = []
        if hedge_qty > 0 and ba is not None:
            orders.append(Order(prod, ba, hedge_qty))
        elif hedge_qty < 0 and bb is not None:
            orders.append(Order(prod, bb, hedge_qty))

        return orders

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Returns (orders_dict, conversions, trader_data_string).
        """
        self._restore(state.traderData)

        # ── Time to expiry (linearly interpolated within the round) ───────────
        T = max(
            TTE_START_DAYS - state.timestamp / TIMESTAMPS_PER_DAY,
            1e-6,
        )

        result: Dict[str, List[Order]] = {}

        # ── 1. HYDROGEL_PACK ──────────────────────────────────────────────────
        result["HYDROGEL_PACK"] = self._trade_hydrogel(state)

        # ── 2. VELVETFRUIT_EXTRACT ────────────────────────────────────────────
        vev_orders, vev_fv = self._trade_vev(state)
        result["VELVETFRUIT_EXTRACT"] = vev_orders

        # ── 3. Vouchers (need VEV fair value as underlying) ───────────────────
        if vev_fv is not None:
            voucher_result = self._trade_vouchers(state, vev_fv, T)
            result.update(voucher_result)

            # ── 4. Optional delta hedge ───────────────────────────────────────
            if DELTA_HEDGE and self.iv_hist:
                window = min(len(self.iv_hist), VOL_SMOOTHING_WINDOW)
                sigma_smooth = sum(self.iv_hist[-window:]) / window
                hedge_orders = self._delta_hedge_orders(
                    state, vev_fv, T, sigma_smooth,
                    result.get("VELVETFRUIT_EXTRACT", []),
                )
                if hedge_orders:
                    result["VELVETFRUIT_EXTRACT"] = (
                        result.get("VELVETFRUIT_EXTRACT", []) + hedge_orders
                    )

        trader_data = self._save()
        conversions  = 0          # no manual conversion needed here

        return result, conversions, trader_data


# ─────────────────────────────────────────────────────────────────────────────
# Tuneable constants (referenced above – kept together for easy tweaking)
# ─────────────────────────────────────────────────────────────────────────────

MAX_HIST = 50   # rolling price history length kept in traderData