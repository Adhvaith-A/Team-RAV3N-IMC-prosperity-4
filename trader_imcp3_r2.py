from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import json
import math


class Trader:
    """IMC Prosperity 3 style trader aligned to round-2 product set.

    This file is intentionally separate from trader.py so IMCP4 tuning remains untouched.
    """

    LIMITS = {
        "SQUID_INK": 50,
        "KELP": 50,
        "RAINFOREST_RESIN": 50,
        "CROISSANTS": 250,
        "JAMS": 350,
        "DJEMBES": 60,
        "PICNIC_BASKET1": 60,
        "PICNIC_BASKET2": 100,
    }

    BASKET_FORMULAS = {
        "PICNIC_BASKET1": {"CROISSANTS": 6.0, "JAMS": 3.0, "DJEMBES": 1.0},
        "PICNIC_BASKET2": {"CROISSANTS": 4.0, "JAMS": 2.0},
    }

    def _clamp(self, x: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, x))

    def _load_data(self, raw: str) -> dict:
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _dump_data(self, data: dict) -> str:
        try:
            return json.dumps(data, separators=(",", ":"))
        except Exception:
            return "{}"

    def _best_bid_ask(self, depth: OrderDepth) -> Tuple[int, int]:
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else 10**9
        return best_bid, best_ask

    def _safe_mid(self, depth: OrderDepth) -> float:
        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid > 0 and best_ask < 10**9:
            return 0.5 * (best_bid + best_ask)
        if best_bid > 0:
            return float(best_bid)
        if best_ask < 10**9:
            return float(best_ask)
        return 0.0

    def _rolling_stats(self, values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 1.0
        n = len(values)
        mean = sum(values) / n
        if n < 2:
            return mean, 1.0
        var = sum((v - mean) * (v - mean) for v in values) / (n - 1)
        return mean, max(1e-6, math.sqrt(var))

    def _append_hist(self, data: dict, key: str, value: float, max_len: int = 240) -> None:
        hist = data.setdefault(key, [])
        hist.append(float(value))
        if len(hist) > max_len:
            del hist[0 : len(hist) - max_len]

    def _target_from_z(self, z: float, limit: int, entry_z: float, deadband_z: float, gain: float) -> int:
        if abs(z) < deadband_z:
            return 0
        if abs(z) < entry_z:
            return 0
        raw = int(round(gain * z * limit))
        return self._clamp(raw, -limit, limit)

    def _execute_delta(self, symbol: str, depth: OrderDepth, delta: int) -> List[Order]:
        orders: List[Order] = []
        if delta == 0:
            return orders

        best_bid, best_ask = self._best_bid_ask(depth)

        if delta > 0:
            # Cross visible ask first for immediate inventory correction.
            cross = 0
            if best_ask < 10**9:
                cross = min(delta, max(0, -depth.sell_orders.get(best_ask, 0)))
                if cross > 0:
                    orders.append(Order(symbol, best_ask, cross))
            rem = delta - cross
            if rem > 0 and best_bid > 0:
                passive_bid = min(best_ask - 1, best_bid + 1) if best_ask < 10**9 else best_bid + 1
                orders.append(Order(symbol, passive_bid, rem))
        else:
            sell_need = -delta
            cross = 0
            if best_bid > 0:
                cross = min(sell_need, max(0, depth.buy_orders.get(best_bid, 0)))
                if cross > 0:
                    orders.append(Order(symbol, best_bid, -cross))
            rem = sell_need - cross
            if rem > 0 and best_ask < 10**9:
                passive_ask = max(best_bid + 1, best_ask - 1) if best_bid > 0 else best_ask - 1
                orders.append(Order(symbol, passive_ask, -rem))

        return orders

    def _quote_mm(self, symbol: str, depth: OrderDepth, pos: int, limit: int, fair: float) -> List[Order]:
        orders: List[Order] = []
        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid <= 0 and best_ask >= 10**9:
            return orders

        spread = max(1.0, float(best_ask - best_bid) if (best_bid > 0 and best_ask < 10**9) else 2.0)
        width = max(1, int(round(0.5 * spread)))
        skew = int(round(0.03 * pos * spread))

        bid_px = int(math.floor(fair - width - skew))
        ask_px = int(math.ceil(fair + width - skew))

        if best_bid > 0:
            bid_px = min(max(1, bid_px), best_bid + 1)
        if best_ask < 10**9:
            ask_px = max(ask_px, best_ask - 1)
        if ask_px <= bid_px:
            ask_px = bid_px + 1

        buy_cap = max(0, limit - pos)
        sell_cap = max(0, limit + pos)
        lot = max(4, min(12, limit // 5))

        if buy_cap > 0:
            orders.append(Order(symbol, bid_px, min(lot, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(symbol, ask_px, -min(lot, sell_cap)))
        return orders

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        orders: Dict[str, List[Order]] = {}

        products = list(self.LIMITS.keys())
        positions = {p: state.position.get(p, 0) for p in products}
        mids: Dict[str, float] = {}

        for p in products:
            depth = state.order_depths.get(p)
            if depth is None:
                continue
            mids[p] = self._safe_mid(depth)

        # Basket spreads against synthetic legs.
        pb_targets = {"PICNIC_BASKET1": 0, "PICNIC_BASKET2": 0}
        spread_cfg = {
            "PICNIC_BASKET1": {"entry": 0.8, "deadband": 0.30, "gain": 0.30},
            "PICNIC_BASKET2": {"entry": 0.9, "deadband": 0.35, "gain": 0.28},
        }

        for pb, formula in self.BASKET_FORMULAS.items():
            if pb not in mids or any(sym not in mids for sym in formula):
                continue

            synth = sum(w * mids[sym] for sym, w in formula.items())
            spread = synth - mids[pb]
            hist_key = f"spread_hist_{pb}"
            self._append_hist(data, hist_key, spread)
            mean, std = self._rolling_stats(data.get(hist_key, []))
            z = (spread - mean) / std

            pb_targets[pb] = self._target_from_z(
                z,
                self.LIMITS[pb],
                spread_cfg[pb]["entry"],
                spread_cfg[pb]["deadband"],
                spread_cfg[pb]["gain"],
            )

        # Build component hedge targets implied by basket targets.
        tgt = dict(positions)
        tgt["PICNIC_BASKET1"] = pb_targets["PICNIC_BASKET1"]
        tgt["PICNIC_BASKET2"] = pb_targets["PICNIC_BASKET2"]
        tgt["CROISSANTS"] = self._clamp(6 * tgt["PICNIC_BASKET1"] + 4 * tgt["PICNIC_BASKET2"], -self.LIMITS["CROISSANTS"], self.LIMITS["CROISSANTS"])
        tgt["JAMS"] = self._clamp(3 * tgt["PICNIC_BASKET1"] + 2 * tgt["PICNIC_BASKET2"], -self.LIMITS["JAMS"], self.LIMITS["JAMS"])
        tgt["DJEMBES"] = self._clamp(tgt["PICNIC_BASKET1"], -self.LIMITS["DJEMBES"], self.LIMITS["DJEMBES"])

        # Execute towards target for round-2 basket complex.
        for sym in ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"]:
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            delta = self._clamp(tgt[sym] - positions[sym], -self.LIMITS[sym], self.LIMITS[sym])
            if delta != 0:
                orders[sym] = self._execute_delta(sym, depth, delta)

        # Round-1 products run as inventory-aware market making.
        for sym in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            depth = state.order_depths.get(sym)
            if depth is None:
                continue

            mid = mids.get(sym, 0.0)
            ema_map = data.setdefault("ema", {})
            prev = ema_map.get(sym, mid)
            fair = 0.85 * prev + 0.15 * mid if mid > 0 else prev
            ema_map[sym] = fair

            mm_orders = self._quote_mm(sym, depth, positions[sym], self.LIMITS[sym], fair)
            if mm_orders:
                orders[sym] = mm_orders

        return orders, 0, self._dump_data(data)
