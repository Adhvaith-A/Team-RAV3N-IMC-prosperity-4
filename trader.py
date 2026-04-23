from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import math
import json
from typing import Any, Optional

import numpy as np
import pandas as pd

#TEAM RAV3N : Adhvaith A and Unnabh M
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
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
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

class Trader:

    def bid(self):
        return 2141

    def _clamp(self, value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _append_iceberg(self, orders: list[Order], symbol: str, price: int, quantity: int, chunk: int = 3):
        if quantity == 0:
            return

        sign = 1 if quantity > 0 else -1
        remaining = abs(quantity)
        while remaining > 0:
            child = min(chunk, remaining)
            orders.append(Order(symbol, price, sign * child))
            remaining -= child

    def _clone_default(self, value):
        if isinstance(value, (dict, list)):
            return json.loads(json.dumps(value))
        return value

    def _window_mean(self, values, window_size: Optional[int] = None) -> float:
        if not values:
            return 0.0
        series = pd.Series(values, dtype="float64")
        if window_size is not None and len(series) > window_size:
            series = series.iloc[-window_size:]
        return float(series.mean())


    # ===== ADVANCED STRATEGY HELPERS (1.2, 2.1, 2.2, 3.1) =====
    
    def _update_volatility_tracking(self, data, symbol, mid_price):
        """Track rolling volatility for dynamic Kalman parameter adjustment (1.2)."""
        if "volatility_window" not in data:
            data["volatility_window"] = {}
        if symbol not in data["volatility_window"]:
            data["volatility_window"][symbol] = []
        
        window = data["volatility_window"][symbol]
        window.append(mid_price)
        
        # Keep only last 20 prices
        if len(window) > 20:
            window.pop(0)
        
        # Calculate rolling volatility
        if len(window) > 2:
            arr = np.asarray(window, dtype=np.float64)
            arr = np.maximum(arr, 1e-9)
            returns = np.log(arr[1:] / arr[:-1])
            mean_return = float(np.mean(returns))
            variance = float(np.var(returns, ddof=1)) if returns.size > 1 else 0.0
            volatility = max(0.0, variance ** 0.5)
        else:
            volatility = 0.01
        
        data["volatility_window"][symbol] = window
        # Also store computed volatility separately for easier access
        if "volatility_values" not in data:
            data["volatility_values"] = {}
        data["volatility_values"][symbol] = volatility
        
        return volatility

    def _update_wavelet_regime(self, data, symbol, mid_price):
        """Estimate chop/trend regime using one-level Haar detail energy."""
        if "wavelet_window" not in data:
            data["wavelet_window"] = {}
        if symbol not in data["wavelet_window"]:
            data["wavelet_window"][symbol] = []

        window = data["wavelet_window"][symbol]
        window.append(float(mid_price))
        if len(window) > self.WAVELET_WINDOW:
            window.pop(0)

        detail_ratio = 0.5
        if len(window) >= 10:
            # Use returns instead of raw prices so low/high frequency energies are scale-invariant.
            arr = np.asarray(window, dtype=np.float64)
            arr = np.maximum(arr, 1e-9)
            returns = np.log(arr[1:] / arr[:-1])

            if returns.size % 2 == 1:
                returns = returns[:-1]

            paired = returns.reshape(-1, 2)
            approximations = 0.5 * (paired[:, 0] + paired[:, 1])
            details = 0.5 * (paired[:, 0] - paired[:, 1])

            detail_energy = float(np.sum(np.square(details)))
            approx_energy = float(np.sum(np.square(approximations)))
            total_energy = max(1e-9, detail_energy + approx_energy)
            detail_ratio = self._clamp(detail_energy / total_energy, 0.0, 1.0)

        if "wavelet_chop" not in data:
            data["wavelet_chop"] = {}
        if "wavelet_chop_ema" not in data:
            data["wavelet_chop_ema"] = {}

        prev_ema = data["wavelet_chop_ema"].get(symbol, detail_ratio)
        chop_ema = 0.78 * prev_ema + 0.22 * detail_ratio
        data["wavelet_chop_ema"][symbol] = chop_ema
        data["wavelet_chop"][symbol] = chop_ema
        return chop_ema

    def _get_dynamic_kalman_params(self, volatility, obi=0.0, trend_strength=0.0, wavelet_chop=0.5, base_q=1e-4, base_r=1e-2):
        """Dynamically adjust Kalman noise parameters based on volatility and liquidity (OBI)."""
        vol_threshold_high = 0.08
        vol_threshold_low = 0.02
        
        if volatility > vol_threshold_high:
            # High volatility: faster adaptation
            Q = base_q * 100  # Increase process noise
            R = base_r * 10   # Increase measurement noise
        elif volatility < vol_threshold_low:
            # Low volatility: more stable
            Q = base_q * 0.1  # Decrease process noise
            R = base_r * 0.1  # Decrease measurement noise
        else:
            # Medium volatility: balanced
            Q = base_q
            R = base_r

        # Liquidity-aware penalty: extreme OBI often means thinner, noisier book.
        obi_threshold = 0.7
        if abs(obi) > obi_threshold:
            liquidity_penalty = self._clamp(1.0 - ((abs(obi) - obi_threshold) / (1.0 - obi_threshold)), 0.25, 1.0)
            Q *= liquidity_penalty
            # Compensate by trusting measurements slightly less.
            R *= (1.0 + (1.0 - liquidity_penalty) * 0.5)

        # Trend-aware / chop-aware adaptation.
        trend_strength = self._clamp(abs(trend_strength), 0.0, 1.0)
        wavelet_chop = self._clamp(wavelet_chop, 0.0, 1.0)
        trend_weight = trend_strength * (1.0 - wavelet_chop)
        chop_weight = wavelet_chop * (1.0 - trend_strength)

        Q *= (1.0 + 0.45 * trend_weight)
        R *= (1.0 - 0.18 * trend_weight)
        Q *= (1.0 - 0.22 * chop_weight)
        R *= (1.0 + 0.32 * chop_weight)

        Q = self._clamp(Q, 1e-6, 0.1)
        R = self._clamp(R, 1e-5, 1.0)
        
        return Q, R

    def _calculate_var_95(self, data, symbol):
        """Calculate 95% VaR from return history for risk parity (2.1)."""
        if "returns_buffer" not in data:
            data["returns_buffer"] = {}
        if symbol not in data["returns_buffer"]:
            data["returns_buffer"][symbol] = []
        
        buffer = data["returns_buffer"][symbol]
        
        # Need at least 20 returns for meaningful VaR
        if len(buffer) < 20:
            return 0.02  # Default 2% VaR
        
        # Calculate 95% VaR (worst 5th percentile)
        sorted_returns = sorted(buffer)
        idx = max(0, int(len(sorted_returns) * 0.05) - 1)
        var_95 = abs(sorted_returns[idx])
        
        return max(0.001, var_95)

    def _update_returns_buffer(self, data, symbol, returns_list, max_len=100):
        """Track returns for VaR calculation (2.1)."""
        if "returns_buffer" not in data:
            data["returns_buffer"] = {}
        if symbol not in data["returns_buffer"]:
            data["returns_buffer"][symbol] = []
        
        buffer = data["returns_buffer"][symbol]
        for ret in returns_list:
            buffer.append(ret)
            if len(buffer) > max_len:
                buffer.pop(0)
        
        data["returns_buffer"][symbol] = buffer

    def _calculate_risk_parity_weights(self, var_osmium, var_pepper):
        """Calculate risk-parity weights for OSMIUM and PEPPER (2.1)."""
        # Inverse volatility weighting
        if var_osmium <= 0.001:
            var_osmium = 0.001
        if var_pepper <= 0.001:
            var_pepper = 0.001
        
        inv_var_osmium = 1.0 / var_osmium
        inv_var_pepper = 1.0 / var_pepper
        total = inv_var_osmium + inv_var_pepper
        
        w_osmium = inv_var_osmium / total
        w_pepper = inv_var_pepper / total
        
        return w_osmium, w_pepper

    def _drawdown_scale(self, data, symbol, threshold=0.08):
        """Return a sizing multiplier based on recent drawdown for a leg (2.1 extension)."""
        returns = data.get("returns_buffer", {}).get(symbol, [])
        if len(returns) < 20:
            return 1.0

        equity = 1.0
        peak = 1.0
        current_dd = 0.0
        # Use recent history to stay reactive while avoiding stale stress states.
        for r in returns[-80:]:
            equity *= (1.0 + r)
            peak = max(peak, equity)
            current_dd = 1.0 - (equity / max(1e-9, peak))

        if current_dd > threshold:
            return 0.5
        # Mild expansion when drawdown is very low keeps utilization high in calm periods.
        if current_dd < 0.02:
            return 1.04
        return 1.0

    def _estimate_correlation(self, data, osmium_price, pepper_price):
        """Estimate correlation between OSMIUM and PEPPER (2.2)."""
        if "prev_osmium_price" not in data:
            data["prev_osmium_price"] = osmium_price
        if "prev_pepper_price" not in data:
            data["prev_pepper_price"] = pepper_price
        
        if "correlation_buffers" not in data:
            data["correlation_buffers"] = {"osmium_rets": [], "pepper_rets": []}
        
        osmium_ret = (osmium_price - data["prev_osmium_price"]) / max(1.0, data["prev_osmium_price"])
        pepper_ret = (pepper_price - data["prev_pepper_price"]) / max(1.0, data["prev_pepper_price"])
        
        data["correlation_buffers"]["osmium_rets"].append(osmium_ret)
        data["correlation_buffers"]["pepper_rets"].append(pepper_ret)
        
        # Keep last 50 returns
        if len(data["correlation_buffers"]["osmium_rets"]) > 50:
            data["correlation_buffers"]["osmium_rets"].pop(0)
            data["correlation_buffers"]["pepper_rets"].pop(0)
        
        # Calculate correlation
        osm_rets = data["correlation_buffers"]["osmium_rets"]
        pepper_rets = data["correlation_buffers"]["pepper_rets"]
        
        if len(osm_rets) > 10:
            osm_mean = sum(osm_rets) / len(osm_rets)
            pepper_mean = sum(pepper_rets) / len(pepper_rets)
            
            cov = sum((osm_rets[i] - osm_mean) * (pepper_rets[i] - pepper_mean) for i in range(len(osm_rets))) / len(osm_rets)
            osm_std = max(1e-6, (sum((r - osm_mean)**2 for r in osm_rets) / len(osm_rets)) ** 0.5)
            pepper_std = max(1e-6, (sum((r - pepper_mean)**2 for r in pepper_rets) / len(pepper_rets)) ** 0.5)
            
            corr = cov / (osm_std * pepper_std)
            corr = self._clamp(corr, -1.0, 1.0)
        else:
            corr = 0.0
        
        data["prev_osmium_price"] = osmium_price
        data["prev_pepper_price"] = pepper_price
        data["correlation_osmium_pepper"] = corr
        
        return corr

    def _dynamic_correlation_threshold(self, volatility):
        """Adaptive hedge trigger: higher threshold in higher-vol regimes (2.2 extension)."""
        base_threshold = 0.7
        # Internal vol units are small; map to a stable multiplier range.
        vol_factor = self._clamp(1.0 + (volatility / 0.02), 1.0, 1.35)
        return self._clamp(base_threshold * vol_factor, 0.7, 0.9)

    def _volatility_position_scale(self, volatility: float) -> float:
        """Reduce exposure when volatility spikes (risk-off sizing)."""
        if self.AGGRESSIVE_EXPERIMENT:
            return 1.0
        if volatility >= self.RISK_OFF_VOL_HARD:
            return 0.58
        if volatility >= self.RISK_OFF_VOL_SOFT:
            return 0.78
        return 1.0

    def _select_osmium_model(self, data) -> str:
        """Dynamically switch OSMIUM model by volatility/trend regime with hysteresis."""
        if self.AGGRESSIVE_EXPERIMENT:
            return "kalman"
        symbol = "ASH_COATED_OSMIUM"
        vol = data.get("volatility_values", {}).get(symbol, 0.005)
        slope_mag = abs(data.get("osmium_slope_ema", {}).get(symbol, 0.0))
        wavelet_chop = data.get("wavelet_chop", {}).get(symbol, 0.5)
        mid = data.get("last_mid_prices", {}).get(symbol, 10000.0)
        fair_dist = abs(mid - 10000.0)

        if "osmium_active_model" not in data:
            data["osmium_active_model"] = {}
        active = data["osmium_active_model"].get(symbol, self.OSMIUM_MODEL)

        low_vol_chop = (
            vol <= self.OSMIUM_SWITCH_LOW_VOL
            and slope_mag < 0.015
            and fair_dist <= 1.6
            and wavelet_chop >= self.WAVELET_CHOP_THRESHOLD
        )
        high_vol_or_trend = (
            vol >= self.OSMIUM_SWITCH_HIGH_VOL
            or slope_mag > 0.035
            or fair_dist > 2.5
            or (wavelet_chop <= self.WAVELET_TREND_THRESHOLD and slope_mag > 0.012)
        )

        if low_vol_chop:
            chosen = "regime"
        elif high_vol_or_trend:
            chosen = "kalman"
        else:
            # Prefer staying in kalman unless all regime conditions are clearly met.
            chosen = "kalman" if active == "kalman" else ("regime" if low_vol_chop else "kalman")

        data["osmium_active_model"][symbol] = chosen
        return chosen

    def __init__(self):
        self.POSITION_LIMITS = {
            "ASH_COATED_OSMIUM": 80,
            "INTARIAN_PEPPER_ROOT": 80
        }
        self.OSMIUM_MODEL = "kalman"
        self.OSMIUM_PROFILE = "mean"
        self.RISK_MULTIPLIER = 8.0
        self.AGGRESSIVE_EXPERIMENT = False
        # Dynamic regime and risk-off sizing knobs.
        self.OSMIUM_SWITCH_LOW_VOL = 0.0018
        self.OSMIUM_SWITCH_HIGH_VOL = 0.0075
        self.RISK_OFF_VOL_SOFT = 0.018
        self.RISK_OFF_VOL_HARD = 0.028
        self.CORR_HEDGE_BETA = 0.35
        self.CORR_HEDGE_BLEND = 0.35
        self.WAVELET_WINDOW = 32
        self.WAVELET_CHOP_THRESHOLD = 0.57
        self.WAVELET_TREND_THRESHOLD = 0.38
        # Advanced strategy tuning
        self.ENABLE_DYNAMIC_KALMAN = True      # (1.2) Dynamic Kalman parameters
        self.ENABLE_RISK_PARITY = not self.AGGRESSIVE_EXPERIMENT         # (2.1) Risk parity allocation
        self.ENABLE_AGGRESSIVE_OBI = True      # (3.1) Aggressive OBI scalping
        self.ENABLE_CORRELATION_HEDGE = not self.AGGRESSIVE_EXPERIMENT   # (2.2) Cross-asset correlation
        self.DEFAULT_STATE = {
            "prev_mid": {},
            "ema_mid": {},
            "ema_ret": {},
            "ema_abs_ret": {},
            "pepper_level": {},
            "pepper_velocity": {},
            "pepper_var": {},
            "osmium_mid_ema": {},
            "osmium_book_ema": {},
            "osmium_slope_ema": {},
            "osmium_mean": {},
            "osmium_var": {},
            "osmium_regime": {},
            "osmium_last_sweep_ts": {},
            "pepper_prev_kalman_gain": {},
            # New state tracking for advanced strategies
            "volatility_window": {},     # Rolling volatility (20-tick window)
            "returns_buffer": {},        # Returns history for VaR
            "correlation_osmium_pepper": 0.0,  # Cross-asset correlation
            "pnl_osmium": {},            # PnL tracking for risk parity
            "pnl_pepper": {},            # PnL tracking for risk parity
            "aggressive_obi_ts": {},     # Last aggressive OBI trigger time
            "osmium_active_model": {},
            "pepper_innovation_ema": {},
            "wavelet_window": {},
            "wavelet_chop": {},
            "wavelet_chop_ema": {},
        }

    def run(self, state: TradingState):
        result = {}
        data = self._restore_state(state.traderData)

        if "last_mid_prices" not in data:
            data["last_mid_prices"] = {}
        # Update cross-asset tracking (1.2, 2.1, 2.2)
        osmium_od = state.order_depths.get("ASH_COATED_OSMIUM")
        pepper_od = state.order_depths.get("INTARIAN_PEPPER_ROOT")
        
        if osmium_od and osmium_od.buy_orders and osmium_od.sell_orders:
            osm_mid = (max(osmium_od.buy_orders.keys()) + min(osmium_od.sell_orders.keys())) / 2.0
            self._update_volatility_tracking(data, "ASH_COATED_OSMIUM", osm_mid)
            self._update_wavelet_regime(data, "ASH_COATED_OSMIUM", osm_mid)
            prev_osm_mid = data["last_mid_prices"].get("ASH_COATED_OSMIUM")
            if prev_osm_mid is not None:
                osm_ret = (osm_mid - prev_osm_mid) / max(1.0, prev_osm_mid)
                self._update_returns_buffer(data, "ASH_COATED_OSMIUM", [osm_ret], max_len=200)
            data["last_mid_prices"]["ASH_COATED_OSMIUM"] = osm_mid
        
        if pepper_od and pepper_od.buy_orders and pepper_od.sell_orders:
            pepper_mid = (max(pepper_od.buy_orders.keys()) + min(pepper_od.sell_orders.keys())) / 2.0
            self._update_volatility_tracking(data, "INTARIAN_PEPPER_ROOT", pepper_mid)
            self._update_wavelet_regime(data, "INTARIAN_PEPPER_ROOT", pepper_mid)
            prev_pepper_mid = data["last_mid_prices"].get("INTARIAN_PEPPER_ROOT")
            if prev_pepper_mid is not None:
                pepper_ret = (pepper_mid - prev_pepper_mid) / max(1.0, prev_pepper_mid)
                self._update_returns_buffer(data, "INTARIAN_PEPPER_ROOT", [pepper_ret], max_len=200)
            data["last_mid_prices"]["INTARIAN_PEPPER_ROOT"] = pepper_mid
            
            # Estimate correlation between assets
            if osmium_od and osmium_od.buy_orders and osmium_od.sell_orders:
                self._estimate_correlation(data, osm_mid, pepper_mid)


        for product in state.order_depths:
            od = state.order_depths[product]
            pos = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 80)
            buy_cap = limit - pos
            sell_cap = -limit - pos

            product_vol = data.get("volatility_values", {}).get(product, 0.0)
            risk_scale = self._volatility_position_scale(product_vol)
            buy_cap = max(0, int(buy_cap * risk_scale))
            sell_cap = -max(0, int(abs(sell_cap) * risk_scale))

            if product == "ASH_COATED_OSMIUM":
                result[product] = self._osmium(od, pos, limit, buy_cap, sell_cap, state.timestamp, data)
            elif product == "INTARIAN_PEPPER_ROOT":
                osmium_pos = state.position.get("ASH_COATED_OSMIUM", 0)
                result[product] = self._pepper(od, pos, limit, buy_cap, sell_cap, state.timestamp, data, osmium_pos)

        trader_data = self._serialize_state(data)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def _restore_state(self, trader_data: str):
        if not trader_data:
            return json.loads(json.dumps(self.DEFAULT_STATE))
        try:
            loaded = json.loads(trader_data)
            for key, default_value in self.DEFAULT_STATE.items():
                if key not in loaded:
                    loaded[key] = self._clone_default(default_value)
                    continue
                # Preserve expected container/scalar type per key.
                if isinstance(default_value, dict) and not isinstance(loaded[key], dict):
                    loaded[key] = self._clone_default(default_value)
                if not isinstance(default_value, dict) and isinstance(loaded[key], dict):
                    loaded[key] = self._clone_default(default_value)
            return loaded
        except Exception:
            return json.loads(json.dumps(self.DEFAULT_STATE))

    def _serialize_state(self, data) -> str:
        try:
            return json.dumps(data)
        except Exception:
            return json.dumps(self.DEFAULT_STATE)

    def _osmium(self, od, pos, limit, buy_cap, sell_cap, timestamp, data):
        active_model = self._select_osmium_model(data)
        if active_model == "regime":
            return self._osmium_regime(od, pos, limit, buy_cap, sell_cap, timestamp, data)
        return self._osmium_kalman(od, pos, limit, buy_cap, sell_cap, timestamp, data)

    def _osmium_kalman(self, od, pos, limit, buy_cap, sell_cap, timestamp, data):
        orders = []
        FAIR = 10000

        profile = {
            "trend_cap": 2.0,
            "trend_gain": 4.0,
            "trend_mid_gain": 0.03,
            "dip_offset": 4,
            "sweep_buy_gap": 2,
            "sweep_sell_gap": 2,
            "inventory_skew": 2.0,
            "signal_obi": 0.8,
            "layer_weight": 0.65,
            "taper_floor": 0.25,
        }
        if self.OSMIUM_PROFILE == "trend":
            profile.update({"trend_cap": 3.0, "trend_gain": 4.6, "trend_mid_gain": 0.035, "dip_offset": 3, "sweep_buy_gap": 1, "sweep_sell_gap": 1, "inventory_skew": 1.8, "signal_obi": 0.9, "layer_weight": 0.6, "taper_floor": 0.2})
        elif self.OSMIUM_PROFILE == "trend_plus":
            profile.update({"trend_cap": 3.5, "trend_gain": 4.9, "trend_mid_gain": 0.04, "dip_offset": 3, "sweep_buy_gap": 1, "sweep_sell_gap": 1, "inventory_skew": 1.6, "signal_obi": 0.95, "layer_weight": 0.58, "taper_floor": 0.15})
        elif self.OSMIUM_PROFILE == "mean":
            profile.update({"trend_cap": 1.5, "trend_gain": 3.4, "trend_mid_gain": 0.025, "dip_offset": 5, "sweep_buy_gap": 2, "sweep_sell_gap": 2, "inventory_skew": 2.2, "signal_obi": 0.7, "layer_weight": 0.7, "taper_floor": 0.3})

        if not od.sell_orders or not od.buy_orders:
            return orders

        best_ask = min(od.sell_orders.keys())
        best_bid = max(od.buy_orders.keys())
        asks_asc = sorted(od.sell_orders.keys())
        bids_desc = sorted(od.buy_orders.keys(), reverse=True)
        mid = (best_ask + best_bid) / 2.0

        if "osmium_mid_ema" not in data:
            data["osmium_mid_ema"] = {}
        if "osmium_book_ema" not in data:
            data["osmium_book_ema"] = {}
        if "osmium_mid_window" not in data:
            data["osmium_mid_window"] = {}
        if "osmium_obi_ema" not in data:
            data["osmium_obi_ema"] = {}

        prev_mid_ema = data["osmium_mid_ema"].get("ASH_COATED_OSMIUM", mid)
        prev_book_ema = data["osmium_book_ema"].get("ASH_COATED_OSMIUM", FAIR)
        osmium_mid_window = data["osmium_mid_window"].get("ASH_COATED_OSMIUM", [])
        osmium_mid_window.append(mid)
        if len(osmium_mid_window) > 18:
            osmium_mid_window.pop(0)
        data["osmium_mid_window"]["ASH_COATED_OSMIUM"] = osmium_mid_window

        buy_vols = np.fromiter(od.buy_orders.values(), dtype=np.float64, count=len(od.buy_orders))
        sell_vols = np.fromiter((abs(v) for v in od.sell_orders.values()), dtype=np.float64, count=len(od.sell_orders))
        total_bid_vol = float(np.sum(buy_vols))
        total_ask_vol = float(np.sum(sell_vols))
        total_vol = max(1, total_bid_vol + total_ask_vol)
        obi = (total_bid_vol - total_ask_vol) / total_vol
        prev_obi_ema = data["osmium_obi_ema"].get("ASH_COATED_OSMIUM", obi)
        osmium_obi_ema = 0.88 * prev_obi_ema + 0.12 * obi
        data["osmium_obi_ema"]["ASH_COATED_OSMIUM"] = osmium_obi_ema

        vol_stress = min(1.0, abs(obi))
        dynamic_limit = max(60, int(limit * (1.0 - 0.25 * vol_stress)))
        buy_cap = min(buy_cap, max(0, dynamic_limit - pos))
        sell_cap = max(sell_cap, min(0, -dynamic_limit - pos))

        mid_ema = 0.25 * mid + 0.75 * prev_mid_ema
        book_ema = 0.40 * (mid + obi * 1.2) + 0.60 * prev_book_ema
        fair_value = int(round(0.7 * FAIR + 0.2 * mid_ema + 0.1 * book_ema))
        slope = mid_ema - prev_mid_ema
        slope_ema_key = "osmium_slope_ema"
        if slope_ema_key not in data:
            data[slope_ema_key] = {}
        prev_slope_ema = data[slope_ema_key].get("ASH_COATED_OSMIUM", 0.0)
        slope_ema = 0.35 * slope + 0.65 * prev_slope_ema

        trend_bias = int(round(max(0.0, min(profile["trend_cap"], (slope_ema * profile["trend_gain"]) + ((mid_ema - FAIR) * profile["trend_mid_gain"])))) )
        fair_value = max(9940, min(10012, fair_value + trend_bias))

        if len(osmium_mid_window) >= 6:
            osmium_ma = sum(osmium_mid_window) / len(osmium_mid_window)
            ma_distance = mid - osmium_ma
            reversion_bias = self._clamp((-(ma_distance / max(1.0, profile["trend_cap"] + 1.0))) - (0.85 * osmium_obi_ema), -2.5, 2.5)
            fair_value = max(9940, min(10012, fair_value + int(round(reversion_bias))))

        data["osmium_mid_ema"]["ASH_COATED_OSMIUM"] = mid_ema
        data["osmium_book_ema"]["ASH_COATED_OSMIUM"] = book_ema
        data[slope_ema_key]["ASH_COATED_OSMIUM"] = slope_ema

        # Dip-buy / rebound-sell overlay: buy only on a real dip, then take profit when price mean-reverts.
        dip_offset = max(3, profile["dip_offset"] - 1)
        if 200 <= timestamp < 5000 and buy_cap > 0:
            dip_signal = mid <= fair_value - dip_offset and slope <= 0 and slope_ema <= prev_slope_ema + 0.01 and obi <= 0.2
            if dip_signal:
                swing_qty = max(1, min(buy_cap, 5))
                swing_price = min(best_ask, fair_value - 1)
                orders.append(Order("ASH_COATED_OSMIUM", swing_price, swing_qty))
                buy_cap -= swing_qty

        if timestamp < 9900 and pos > 0 and sell_cap < 0:
            rebound_signal = mid >= fair_value - 1 and slope_ema >= -0.01
            if rebound_signal:
                take_profit_qty = max(1, min(pos, abs(sell_cap), 4))
                take_profit_price = max(best_bid, fair_value + 1)
                orders.append(Order("ASH_COATED_OSMIUM", take_profit_price, -take_profit_qty))
                sell_cap += take_profit_qty

        # 1. Unbounded Sweep-Taker
        # Aggressively cross the spread when the book is far from the adaptive fair.
        if "osmium_last_sweep_ts" not in data:
            data["osmium_last_sweep_ts"] = {}
        last_sweep = data["osmium_last_sweep_ts"].get("ASH_COATED_OSMIUM", -10_000)
        sweep_fired = False

        # ADVANCED (1.2): Adjust sweep thresholds based on volatility.
        osm_vol = data.get("volatility_values", {}).get("ASH_COATED_OSMIUM", 0.005)
        if osm_vol < 0.003:
            sweep_obi_threshold = 0.4
            sweep_cooldown = 800
        elif osm_vol > 0.015:
            sweep_obi_threshold = 0.6
            sweep_cooldown = 1200
        else:
            sweep_obi_threshold = 0.46
            sweep_cooldown = 900

        # 20x risk mode: trigger sweeps much more easily and frequently.
        risk = max(1.0, self.RISK_MULTIPLIER)
        sweep_obi_threshold = self._clamp(sweep_obi_threshold / risk, 0.02, 1.0)
        sweep_cooldown = max(1, int(sweep_cooldown / risk))

        sweep_enabled = abs(obi) >= sweep_obi_threshold and (timestamp - last_sweep >= sweep_cooldown)
        if timestamp < 400:
            sweep_enabled = False

        if buy_cap > 0 and sweep_enabled:
            for ask_level in asks_asc:
                if ask_level <= fair_value - profile["sweep_buy_gap"] or (ask_level <= fair_value - 1 and pos < 60):
                    avail = abs(od.sell_orders[ask_level])
                    qty = min(avail, buy_cap)
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", ask_level, qty))
                        buy_cap -= qty
                        sweep_fired = True
                    if buy_cap <= 0:
                        break

        if sell_cap < 0 and sweep_enabled:
            for bid_level in bids_desc:
                if bid_level >= fair_value + profile["sweep_sell_gap"] or (bid_level >= fair_value + 1 and pos > -60):
                    avail = od.buy_orders[bid_level]
                    qty = min(avail, abs(sell_cap))
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", bid_level, -qty))
                        sell_cap += qty
                        sweep_fired = True
                    if sell_cap >= 0:
                        break

        if sweep_fired:
            data["osmium_last_sweep_ts"]["ASH_COATED_OSMIUM"] = timestamp

        # 2. Adaptive fair value with inventory-aware skew.
        c = pos / float(limit)
        inventory_skew = c * profile["inventory_skew"]
        reservation_price = fair_value - inventory_skew

        signal = reservation_price + (obi * profile["signal_obi"])
        base_bid = min(int(round(signal - 1)), best_ask - 1)
        base_ask = max(int(round(signal + 1)), best_bid + 1)
        if base_bid >= base_ask:
            base_bid = min(base_bid, base_ask - 1)

        # 3. Multi-layer maker ladder with volatility-aware weighting.
        if buy_cap > 0:
            taper = max(profile["taper_floor"], 1.0 - max(0.0, pos - 70) / 20.0)
            layer_weight = profile["layer_weight"]
            if osm_vol > 0.010:
                layer_weight = min(0.9, layer_weight + 0.05)
            elif osm_vol < 0.003:
                layer_weight = max(0.5, layer_weight - 0.05)
            qty_layer1 = max(1, int(buy_cap * (layer_weight * taper)))
            qty_layer2 = buy_cap - qty_layer1
            orders.append(Order("ASH_COATED_OSMIUM", base_bid, qty_layer1))
            if qty_layer2 > 0 and base_bid - 1 > 0:
                orders.append(Order("ASH_COATED_OSMIUM", base_bid - 1, qty_layer2))

        if sell_cap < 0:
            abs_sell_cap = abs(sell_cap)
            taper = max(profile["taper_floor"], 1.0 - max(0.0, -pos - 70) / 20.0)
            layer_weight = profile["layer_weight"]
            if osm_vol > 0.010:
                layer_weight = min(0.9, layer_weight + 0.05)
            elif osm_vol < 0.003:
                layer_weight = max(0.5, layer_weight - 0.05)
            qty_layer1 = max(1, int(abs_sell_cap * (layer_weight * taper)))
            qty_layer2 = abs_sell_cap - qty_layer1
            orders.append(Order("ASH_COATED_OSMIUM", base_ask, -qty_layer1))
            if qty_layer2 > 0:
                orders.append(Order("ASH_COATED_OSMIUM", base_ask + 1, -qty_layer2))

        return orders

    def _osmium_regime(self, od, pos, limit, buy_cap, sell_cap, timestamp, data):
        orders = []
        FAIR = 10000

        if not od.sell_orders or not od.buy_orders:
            return orders

        best_ask = min(od.sell_orders.keys())
        best_bid = max(od.buy_orders.keys())
        asks_asc = sorted(od.sell_orders.keys())
        bids_desc = sorted(od.buy_orders.keys(), reverse=True)
        mid = (best_ask + best_bid) / 2.0

        if "osmium_mean" not in data:
            data["osmium_mean"] = {}
        if "osmium_var" not in data:
            data["osmium_var"] = {}
        if "osmium_regime" not in data:
            data["osmium_regime"] = {}

        prev_mean = data["osmium_mean"].get("ASH_COATED_OSMIUM", mid)
        prev_var = data["osmium_var"].get("ASH_COATED_OSMIUM", 4.0)
        prev_regime = data["osmium_regime"].get("ASH_COATED_OSMIUM", 0.0)

        buy_vols = np.fromiter(od.buy_orders.values(), dtype=np.float64, count=len(od.buy_orders))
        sell_vols = np.fromiter((abs(v) for v in od.sell_orders.values()), dtype=np.float64, count=len(od.sell_orders))
        total_bid_vol = float(np.sum(buy_vols))
        total_ask_vol = float(np.sum(sell_vols))
        total_vol = max(1, total_bid_vol + total_ask_vol)
        obi = (total_bid_vol - total_ask_vol) / total_vol

        ret = mid - prev_mean
        regime_score = 0.65 * prev_regime + 0.35 * (ret + obi * 1.5)
        variance = 0.9 * prev_var + 0.1 * (ret * ret)
        mean_reversion_band = max(2.0, min(6.0, (variance ** 0.5) * 1.3))

        data["osmium_mean"]["ASH_COATED_OSMIUM"] = 0.18 * mid + 0.82 * prev_mean
        data["osmium_var"]["ASH_COATED_OSMIUM"] = variance
        data["osmium_regime"]["ASH_COATED_OSMIUM"] = regime_score

        if abs(regime_score) >= 1.2:
            fair_value = int(round(mid + max(-4.0, min(4.0, regime_score * 1.4))))
        else:
            fair_value = int(round(0.85 * FAIR + 0.15 * prev_mean))

        fair_value = max(9990, min(10012, fair_value))

        if timestamp < 9800 and buy_cap > 0 and regime_score <= 0.6 and mid <= fair_value - mean_reversion_band:
            swing_qty = max(1, min(buy_cap, 6))
            orders.append(Order("ASH_COATED_OSMIUM", min(best_ask, fair_value - 1), swing_qty))
            buy_cap -= swing_qty

        if timestamp < 9800 and sell_cap < 0 and regime_score >= -0.6 and mid >= fair_value + mean_reversion_band:
            swing_qty = max(1, min(abs(sell_cap), 6))
            orders.append(Order("ASH_COATED_OSMIUM", max(best_bid, fair_value + 1), -swing_qty))
            sell_cap += swing_qty

        if buy_cap > 0:
            if mid <= fair_value - 2 or (regime_score > 1.0 and mid <= fair_value):
                for ask_level in asks_asc:
                    if ask_level <= fair_value + 1:
                        avail = abs(od.sell_orders[ask_level])
                        qty = min(avail, buy_cap)
                        if qty > 0:
                            orders.append(Order("ASH_COATED_OSMIUM", ask_level, qty))
                            buy_cap -= qty

        if sell_cap < 0:
            if mid >= fair_value + 2 or (regime_score < -1.0 and mid >= fair_value):
                for bid_level in bids_desc:
                    if bid_level >= fair_value - 1:
                        avail = od.buy_orders[bid_level]
                        qty = min(avail, abs(sell_cap))
                        if qty > 0:
                            orders.append(Order("ASH_COATED_OSMIUM", bid_level, -qty))
                            sell_cap += qty

        c = pos / float(limit)
        inventory_skew = c * 2.2
        if abs(regime_score) >= 1.0:
            reservation_price = fair_value + (0.4 * regime_score) - inventory_skew
        else:
            reservation_price = fair_value - inventory_skew

        signal = reservation_price + (obi * 0.6)
        base_bid = min(int(round(signal - 1)), best_ask - 1)
        base_ask = max(int(round(signal + 1)), best_bid + 1)
        if base_bid >= base_ask:
            base_bid = min(base_bid, base_ask - 1)

        if buy_cap > 0:
            taper = max(0.2, 1.0 - max(0.0, pos - 70) / 20.0)
            qty_layer1 = max(1, int(buy_cap * (0.55 * taper)))
            qty_layer2 = buy_cap - qty_layer1
            orders.append(Order("ASH_COATED_OSMIUM", base_bid, qty_layer1))
            if qty_layer2 > 0 and base_bid - 1 > 0:
                orders.append(Order("ASH_COATED_OSMIUM", base_bid - 1, qty_layer2))

        if sell_cap < 0:
            abs_sell_cap = abs(sell_cap)
            taper = max(0.2, 1.0 - max(0.0, -pos - 70) / 20.0)
            qty_layer1 = max(1, int(abs_sell_cap * (0.55 * taper)))
            qty_layer2 = abs_sell_cap - qty_layer1
            orders.append(Order("ASH_COATED_OSMIUM", base_ask, -qty_layer1))
            if qty_layer2 > 0:
                orders.append(Order("ASH_COATED_OSMIUM", base_ask + 1, -qty_layer2))

        return orders

    def _pepper(self, od, pos, limit, buy_cap, sell_cap, timestamp, data, osmium_pos=0):
        orders = []

        if not od.sell_orders or not od.buy_orders:
            return orders

        best_ask = min(od.sell_orders.keys())
        best_bid = max(od.buy_orders.keys())
        asks_asc = sorted(od.sell_orders.keys())
        bids_desc = sorted(od.buy_orders.keys(), reverse=True)
        mid = (best_ask + best_bid) / 2.0

        warmup_scale = 1.0
        if timestamp < 600 and not self.AGGRESSIVE_EXPERIMENT:
            warmup_scale = 0.25 + (0.75 * (timestamp / 600.0))
            buy_cap = max(0, int(buy_cap * warmup_scale))
            sell_cap = -max(0, int(abs(sell_cap) * warmup_scale))

        if "fast_ema" not in data:
            data["fast_ema"] = {}
        if "slow_ema" not in data:
            data["slow_ema"] = {}
        if "pepper_mid_window" not in data:
            data["pepper_mid_window"] = {}
        if "pepper_obi_ema" not in data:
            data["pepper_obi_ema"] = {}
        if "pepper_orderflow_ema" not in data:
            data["pepper_orderflow_ema"] = {}

        fast_ema = data["fast_ema"].get("INTARIAN_PEPPER_ROOT", mid)
        slow_ema = data["slow_ema"].get("INTARIAN_PEPPER_ROOT", mid)
        mid_window = data["pepper_mid_window"].get("INTARIAN_PEPPER_ROOT", [])
        mid_window.append(mid)
        if len(mid_window) > 30:
            mid_window.pop(0)
        data["pepper_mid_window"]["INTARIAN_PEPPER_ROOT"] = mid_window
        pepper_level = data.get("pepper_level", {}).get("INTARIAN_PEPPER_ROOT", mid)
        pepper_velocity = data.get("pepper_velocity", {}).get("INTARIAN_PEPPER_ROOT", 0.0)
        pepper_var = data.get("pepper_var", {}).get("INTARIAN_PEPPER_ROOT", 4.0)

        # 1. Signal Generation: Tuned EMA periods for trend quality.
        fast_alpha = 2.0 / (12.0 + 1.0)
        slow_alpha = 2.0 / (60.0 + 1.0)
        fast_ema = fast_alpha * mid + (1.0 - fast_alpha) * fast_ema
        slow_ema = slow_alpha * mid + (1.0 - slow_alpha) * slow_ema
        alpha_signal = fast_ema - slow_ema

        # 2. Cumulative Order Book Imbalance (Acceleration)
        total_bid_vol = sum(od.buy_orders.values())
        total_ask_vol = sum(abs(v) for v in od.sell_orders.values())
        total_vol = max(1, total_bid_vol + total_ask_vol)
        obi = (total_bid_vol - total_ask_vol) / total_vol
        prev_pepper_obi_ema = data["pepper_obi_ema"].get("INTARIAN_PEPPER_ROOT", obi)
        pepper_obi_ema = 0.86 * prev_pepper_obi_ema + 0.14 * obi
        data["pepper_obi_ema"]["INTARIAN_PEPPER_ROOT"] = pepper_obi_ema

        # Use top-10 levels for depth pressure and rebalance blend under volatility.
        buy_prices = np.asarray(sorted(od.buy_orders.keys(), reverse=True)[:10], dtype=np.float64)
        buy_depth = np.asarray([od.buy_orders[price] for price in buy_prices], dtype=np.float64)
        sell_prices = np.asarray(sorted(od.sell_orders.keys())[:10], dtype=np.float64)
        sell_depth = np.asarray([abs(od.sell_orders[price]) for price in sell_prices], dtype=np.float64)
        deep_bid_vol = float(np.sum(buy_depth / np.maximum(1.0, mid - buy_prices)))
        deep_ask_vol = float(np.sum(sell_depth / np.maximum(1.0, sell_prices - mid)))

        deep_total = max(1.0, deep_bid_vol + deep_ask_vol)
        deep_obi = (deep_bid_vol - deep_ask_vol) / deep_total
        vol_factor = self._clamp(math.sqrt(max(1.0, pepper_var)) / 2.5, 0.0, 1.0)
        depth_weight = self._clamp(0.40 + 0.30 * vol_factor, 0.40, 0.70)
        top_weight = 1.0 - depth_weight
        blended_obi = max(-1.0, min(1.0, top_weight * obi + depth_weight * deep_obi))

        # Kalman-style uncertainty tracking: keep the main drift signal intact, but widen risk controls when the estimate is noisy.
        measured_fair = mid + (blended_obi * 0.7) + (alpha_signal * 0.08)
        measurement_var = self._clamp((3.4 + (1.8 * vol_factor)) - abs(blended_obi), 1.2, 6.0)
        innovation = measured_fair - pepper_level

        # ADVANCED (1.2): Dynamically adjust Kalman parameters based on volatility
        Q = 1e-4
        R = 1e-2
        wavelet_chop = data.get("wavelet_chop", {}).get("INTARIAN_PEPPER_ROOT", 0.5)
        if self.ENABLE_DYNAMIC_KALMAN:
            pepper_vol = data.get("volatility_values", {}).get("INTARIAN_PEPPER_ROOT", 0.01)
            trend_strength = self._clamp(abs(alpha_signal) / 2.0, 0.0, 1.0)
            Q, R = self._get_dynamic_kalman_params(
                pepper_vol,
                blended_obi,
                trend_strength=trend_strength,
                wavelet_chop=wavelet_chop,
            )
            # Modulate measurement_var dynamically: high vol -> higher uncertainty
            vol_multiplier = 1.0 + (2.0 * pepper_vol / 0.1)  # Scale to ~3x at high vol
            measurement_var = self._clamp(measurement_var * vol_multiplier, 1.2, 8.0)

        process_var = pepper_var + (Q * 2500.0)
        adjusted_measurement_var = measurement_var + (R * 40.0)
        raw_kalman_gain = process_var / max(1e-6, (process_var + adjusted_measurement_var))
        prev_kalman_gain = data.get("pepper_prev_kalman_gain", {}).get("INTARIAN_PEPPER_ROOT", raw_kalman_gain)
        kalman_gain = self._clamp(0.7 * prev_kalman_gain + 0.3 * raw_kalman_gain, 0.1, 0.9)
        if "pepper_innovation_ema" not in data:
            data["pepper_innovation_ema"] = {}
        prev_innov_ema = data["pepper_innovation_ema"].get("INTARIAN_PEPPER_ROOT", abs(innovation))
        innovation_ema = 0.85 * prev_innov_ema + 0.15 * abs(innovation)
        data["pepper_innovation_ema"]["INTARIAN_PEPPER_ROOT"] = innovation_ema
        pepper_level = pepper_level + (kalman_gain * innovation)
        pepper_velocity = 0.9 * pepper_velocity + 0.1 * innovation
        pepper_var = max(1.0, 0.92 * pepper_var + 0.08 * (innovation * innovation))
        # 3. Micro-Price Drift & Target Position
        # Combine trend velocity with momentary book acceleration.
        predicted_drift = (alpha_signal * 0.9) + (blended_obi * 1.55)
        predicted_drift *= self.RISK_MULTIPLIER

        if "alpha_abs_ema" not in data:
            data["alpha_abs_ema"] = {}
        alpha_abs = data["alpha_abs_ema"].get("INTARIAN_PEPPER_ROOT", abs(alpha_signal))
        alpha_abs = 0.9 * alpha_abs + 0.1 * abs(alpha_signal)
        data["alpha_abs_ema"]["INTARIAN_PEPPER_ROOT"] = alpha_abs

        sigma = max(0.1, alpha_abs)
        if measurement_var >= 3.6:
            hedge_trigger_mult = 1.3
        elif measurement_var <= 2.0:
            hedge_trigger_mult = 1.7
        else:
            hedge_trigger_mult = 1.5
        if abs(alpha_signal) > hedge_trigger_mult * sigma:
            hedge_ratio = -0.2 * (osmium_pos / float(max(1, limit)))
            predicted_drift += hedge_ratio

        trend_bias = 0.0
        if len(mid_window) >= 6:
            short_ma = self._window_mean(mid_window, 6)
            long_ma = self._window_mean(mid_window)
            slope = mid_window[-1] - mid_window[-6]
            range_high = max(mid_window)
            range_low = min(mid_window)
            range_width = max(1.0, range_high - range_low)
            breakout_up = (mid >= range_high - 0.25 * range_width) and slope > 0
            breakout_down = (mid <= range_low + 0.25 * range_width) and slope < 0
            trend_bias = self._clamp(((short_ma - long_ma) / max(1.0, long_ma)) * 90.0 + (slope / max(1.0, long_ma)) * 180.0, -1.0, 1.0)
            if breakout_up:
                trend_bias = max(trend_bias, 0.35)
            elif breakout_down:
                trend_bias = min(trend_bias, -0.35)

        if trend_bias > 0.15:
            predicted_drift += 0.45 * trend_bias
        elif trend_bias < -0.15:
            predicted_drift += 0.45 * trend_bias

        orderflow_signal = 0.0
        if len(mid_window) >= 4:
            short_ret = (mid - mid_window[-4]) / max(1.0, mid_window[-4])
            orderflow_signal += self._clamp(short_ret * 160.0, -0.8, 0.8)
        if len(mid_window) >= 10:
            ma_10 = self._window_mean(mid_window, 10)
            ma_distance = (mid - ma_10) / max(1.0, ma_10)
            orderflow_signal += self._clamp((-ma_distance) * 85.0, -0.7, 0.7)
        orderflow_signal += self._clamp((pepper_obi_ema - prev_pepper_obi_ema) * 1.6 + pepper_obi_ema * 0.4, -0.9, 0.9)
        orderflow_signal = self._clamp(orderflow_signal, -1.5, 1.5)
        data["pepper_orderflow_ema"]["INTARIAN_PEPPER_ROOT"] = 0.82 * data["pepper_orderflow_ema"].get("INTARIAN_PEPPER_ROOT", orderflow_signal) + 0.18 * orderflow_signal
        predicted_drift += 0.18 * data["pepper_orderflow_ema"]["INTARIAN_PEPPER_ROOT"]
        
        # Continuous target positional bias using tanh limit
        target_pos = int(round(limit * math.tanh(predicted_drift / 2.0)))
        if timestamp < 600:
            ramp_cap = max(10, int(limit * warmup_scale))
            target_pos = max(-ramp_cap, min(ramp_cap, target_pos))
        if trend_bias > 0.2:
            target_pos = max(target_pos, 0)
        elif trend_bias < -0.2:
            target_pos = min(target_pos, 0)
        if timestamp >= 9950:
            target_pos = 0 # Unwind into the close safely

        hedge_target = None
        corr = data.get("correlation_osmium_pepper", 0.0)
        corr_threshold = self._dynamic_correlation_threshold(pepper_vol)
        if self.ENABLE_CORRELATION_HEDGE and corr > corr_threshold and abs(osmium_pos) >= 20:
            hedge_target = int(round(-self.CORR_HEDGE_BETA * osmium_pos))
            hedge_target = max(-limit, min(limit, hedge_target))
            target_pos = int(round((1.0 - self.CORR_HEDGE_BLEND) * target_pos + self.CORR_HEDGE_BLEND * hedge_target))

        # Calculate safety-anchored fair value incorporating inventory skew
        inventory_skew = (pos / float(limit)) * 1.5
        reservation_price = mid + predicted_drift - inventory_skew

        # Baseline maker quoting distances
        spread = self._clamp(1.0 + min(0.5, math.sqrt(pepper_var) * 0.06), 1.0, 1.5)
        # Let regime signal influence quoting: wider in chop, tighter in directional tape.
        spread_regime_scale = self._clamp(0.90 + 0.24 * wavelet_chop, 0.88, 1.12)
        spread = self._clamp(spread * spread_regime_scale, 0.9, 1.55)
        base_bid = int(round(reservation_price - spread))
        base_ask = int(round(reservation_price + spread))
        # Hard limits to never cross the spread passively
        base_bid = min(base_bid, best_ask - 1)
        base_ask = max(base_ask, best_bid + 1)

        # 4. Asymmetric Taker Protocol
        # Only cross the spread aggressively if the trend momentum completely overtakes the spread cost
        taker_threshold = self._clamp(1.5 + (1.0 - kalman_gain) * 0.35, 1.5, 1.8)
        if timestamp < 600 and not self.AGGRESSIVE_EXPERIMENT:
            taker_threshold *= 1.35
        if not self.AGGRESSIVE_EXPERIMENT:
            lag_ratio = innovation_ema / max(1.0, measurement_var)
            taker_threshold *= self._clamp(1.0 + (0.22 * lag_ratio), 1.0, 1.35)
        # In choppy regimes require stronger confirmation; in trending regimes react sooner.
        taker_threshold *= self._clamp(0.90 + 0.34 * wavelet_chop, 0.88, 1.20)
        taker_threshold = self._clamp(taker_threshold / max(1.0, self.RISK_MULTIPLIER), 0.05, 2.0)
        if self.AGGRESSIVE_EXPERIMENT:
            taker_threshold = min(taker_threshold, 0.03)

        # Volatility filter for OBI scalping: operate only in a medium-volatility band.
        pepper_vol = data.get("volatility_values", {}).get("INTARIAN_PEPPER_ROOT", 0.01)
        annualized_vol_proxy = pepper_vol * 55.0
        should_scalp_obi = 0.0 < annualized_vol_proxy < 10.0 and 1.4 <= measurement_var <= 4.3 and kalman_gain >= 0.22
        if self.AGGRESSIVE_EXPERIMENT:
            should_scalp_obi = True

        # ADVANCED (3.1): Aggressive OBI scalping on smaller imbalances (30-50%)
        if self.ENABLE_AGGRESSIVE_OBI and should_scalp_obi and timestamp >= 600:
            if "aggressive_obi_ts" not in data:
                data["aggressive_obi_ts"] = {}
            last_aggressive = data["aggressive_obi_ts"].get("INTARIAN_PEPPER_ROOT", -5000)
            
            # Scalp smaller imbalances with tighter stops
            if abs(blended_obi) >= 0.32 and abs(blended_obi) < 0.50 and (timestamp - last_aggressive >= 900):
                if blended_obi > 0.32 and buy_cap > 0:
                    # Bid aggression on buy imbalance
                    for ask_level in asks_asc[:5]:  # Top 5 asks
                        if ask_level <= reservation_price:
                            avail = abs(od.sell_orders[ask_level])
                            qty = min(avail, max(1, buy_cap // 3))  # 1/3 of capacity
                            if qty > 0:
                                self._append_iceberg(orders, "INTARIAN_PEPPER_ROOT", ask_level, qty, chunk=2)
                                buy_cap -= qty
                                data["aggressive_obi_ts"]["INTARIAN_PEPPER_ROOT"] = timestamp
                
                elif blended_obi < -0.32 and sell_cap < 0:
                    # Ask aggression on sell imbalance
                    for bid_level in bids_desc[:5]:  # Top 5 bids
                        if bid_level >= reservation_price:
                            avail = od.buy_orders[bid_level]
                            qty = min(avail, max(1, abs(sell_cap) // 3))  # 1/3 of capacity
                            if qty > 0:
                                self._append_iceberg(orders, "INTARIAN_PEPPER_ROOT", bid_level, -qty, chunk=2)
                                sell_cap += qty
                                data["aggressive_obi_ts"]["INTARIAN_PEPPER_ROOT"] = timestamp

        
        hedge_buy_urgent = hedge_target is not None and corr > corr_threshold and pos < hedge_target and osmium_pos < -25 and measurement_var <= 4.5
        hedge_sell_urgent = hedge_target is not None and corr > corr_threshold and pos > hedge_target and osmium_pos > 25 and measurement_var <= 4.5

        if buy_cap > 0 and (predicted_drift > taker_threshold or hedge_buy_urgent):
            for ask_level in asks_asc:
                # Only take if we are buying cheaper than our true anticipated future price
                cross_for_alpha = ask_level <= reservation_price and ask_level <= (reservation_price - (0.3 * taker_threshold))
                cross_for_hedge = hedge_buy_urgent and ask_level <= best_ask + 1
                if cross_for_alpha or cross_for_hedge:
                    avail = abs(od.sell_orders[ask_level])
                    qty_cap = buy_cap if cross_for_alpha else max(1, buy_cap // 2)
                    qty = min(avail, qty_cap)
                    if qty > 0:
                        self._append_iceberg(orders, "INTARIAN_PEPPER_ROOT", ask_level, qty, chunk=4)
                        buy_cap -= qty
                    if buy_cap <= 0:
                        break

        if sell_cap < 0 and (predicted_drift < -taker_threshold or hedge_sell_urgent):
            for bid_level in bids_desc:
                # Only short if the bid is richer than our true anticipated future price
                cross_for_alpha = bid_level >= reservation_price and bid_level >= (reservation_price + (0.3 * taker_threshold))
                cross_for_hedge = hedge_sell_urgent and bid_level >= best_bid - 1
                if cross_for_alpha or cross_for_hedge:
                    avail = od.buy_orders[bid_level]
                    qty_cap = abs(sell_cap) if cross_for_alpha else max(1, abs(sell_cap) // 2)
                    qty = min(avail, qty_cap)
                    if qty > 0:
                        self._append_iceberg(orders, "INTARIAN_PEPPER_ROOT", bid_level, -qty, chunk=4)
                        sell_cap += qty
                    if sell_cap >= 0:
                        break

        if "osmium_slope_ema" not in data:
            data["osmium_slope_ema"] = {}
        osm_slope = data["osmium_slope_ema"].get("ASH_COATED_OSMIUM", 0.0)
        max_position = min(limit, 80 + int(10 * (1 - abs(osm_slope))))
        uncertainty_cut = self._clamp(1.0 - (measurement_var / 20.0), 0.55, 1.0)
        if self.AGGRESSIVE_EXPERIMENT:
            uncertainty_cut = 1.0
        effective_limit = max(20, int(max_position * uncertainty_cut))
        buy_cap = min(buy_cap, max(0, effective_limit - pos))
        sell_cap = max(sell_cap, min(0, -effective_limit - pos))

        # ADVANCED (2.1 & 2.2): Risk parity and correlation hedging
        if self.ENABLE_RISK_PARITY or self.ENABLE_CORRELATION_HEDGE:
            # Calculate VaR for risk parity weighting
            var_pepper = self._calculate_var_95(data, "INTARIAN_PEPPER_ROOT")
            var_osmium = self._calculate_var_95(data, "ASH_COATED_OSMIUM")

            if self.ENABLE_RISK_PARITY:
                w_osmium, w_pepper = self._calculate_risk_parity_weights(var_osmium, var_pepper)
                # Adjust effective limit based on risk parity weight
                # If pepper is riskier, reduce its limit slightly
                risk_parity_factor = self._clamp(w_pepper, 0.95, 1.05)
                # Drawdown-aware rebalance: cut leg exposure under stress.
                dd_scale_pepper = self._drawdown_scale(data, "INTARIAN_PEPPER_ROOT", threshold=0.08)
                risk_parity_factor *= dd_scale_pepper
                effective_limit = max(20, int(effective_limit * risk_parity_factor))

            if self.ENABLE_CORRELATION_HEDGE:
                corr = data.get("correlation_osmium_pepper", 0.0)
                pepper_vol = data.get("volatility_values", {}).get("INTARIAN_PEPPER_ROOT", 0.01)
                corr_threshold = self._dynamic_correlation_threshold(pepper_vol)
                # If correlation > 0.7, reduce position (hedge against systemic moves)
                if abs(corr) > corr_threshold:
                    hedge_factor = 0.85  # Reduce position by 15%
                    effective_limit = max(20, int(effective_limit * hedge_factor))
                    # If positive correlation and osmium long, reduce pepper long
                    if corr > corr_threshold and osmium_pos > 30:
                        effective_limit = max(10, int(effective_limit * 0.8))

        # Reapply position limits after advanced adjustments
        buy_cap = min(buy_cap, max(0, effective_limit - pos))
        sell_cap = max(sell_cap, min(0, -effective_limit - pos))

        if vol_factor >= 0.7:
            stop_mult = 1.8
        elif vol_factor <= 0.35:
            stop_mult = 2.2
        else:
            stop_mult = 2.0
        stop_mult *= self.RISK_MULTIPLIER
        if self.AGGRESSIVE_EXPERIMENT:
            stop_mult = 100.0
        stop_long = reservation_price - (stop_mult * measurement_var)
        stop_short = reservation_price + (stop_mult * measurement_var)
        if pos > 0 and mid < stop_long and sell_cap < 0:
            stop_qty = min(pos, abs(sell_cap))
            if stop_qty > 0:
                self._append_iceberg(orders, "INTARIAN_PEPPER_ROOT", best_bid, -stop_qty, chunk=4)
                sell_cap += stop_qty
        if pos < 0 and mid > stop_short and buy_cap > 0:
            stop_qty = min(abs(pos), buy_cap)
            if stop_qty > 0:
                self._append_iceberg(orders, "INTARIAN_PEPPER_ROOT", best_ask, stop_qty, chunk=4)
                buy_cap -= stop_qty

        # 5. Inventory-Aware Maker Replenishment (No Blind Pennying)
        if buy_cap > 0:
            if target_pos > pos:
                # We actively want to increase position. Quote up to base_bid, which includes drift.
                # In strong uptrends, this inherently pennies safely.
                optimal_bid = min(best_bid + 1, base_bid)
            else:
                # We do not want to increase position. Step back and fade the market.
                optimal_bid = min(best_bid - 1, base_bid)
                
            layer1_frac = self._clamp(0.72 - 0.12 * vol_factor, 0.60, 0.72)
            qty_layer1 = max(1, int(buy_cap * layer1_frac))
            qty_layer2 = buy_cap - qty_layer1
            if trend_bias > 0.2:
                qty_layer1 = max(1, int(qty_layer1 * 0.5))
            self._append_iceberg(orders, "INTARIAN_PEPPER_ROOT", optimal_bid, qty_layer1, chunk=3)
            if qty_layer2 > 0 and optimal_bid - 1 > 0:
                self._append_iceberg(orders, "INTARIAN_PEPPER_ROOT", optimal_bid - 1, qty_layer2, chunk=3)

        if sell_cap < 0:
            abs_sell_cap = abs(sell_cap)
            if target_pos < pos:
                # We actively want to decrease position / short
                optimal_ask = max(best_ask - 1, base_ask)
            else:
                # We want to hold longs, fade the ask
                optimal_ask = max(best_ask + 1, base_ask)

            layer1_frac = self._clamp(0.72 - 0.12 * vol_factor, 0.60, 0.72)
            qty_layer1 = max(1, int(abs_sell_cap * layer1_frac))
            qty_layer2 = abs_sell_cap - qty_layer1
            if trend_bias < -0.2:
                qty_layer1 = max(1, int(qty_layer1 * 0.5))
            self._append_iceberg(orders, "INTARIAN_PEPPER_ROOT", optimal_ask, -qty_layer1, chunk=3)
            if qty_layer2 > 0:
                self._append_iceberg(orders, "INTARIAN_PEPPER_ROOT", optimal_ask + 1, -qty_layer2, chunk=3)

        # 6. Save Persistent State
        data["fast_ema"]["INTARIAN_PEPPER_ROOT"] = fast_ema
        data["slow_ema"]["INTARIAN_PEPPER_ROOT"] = slow_ema
        data["pepper_level"]["INTARIAN_PEPPER_ROOT"] = pepper_level
        data["pepper_velocity"]["INTARIAN_PEPPER_ROOT"] = pepper_velocity
        data["pepper_var"]["INTARIAN_PEPPER_ROOT"] = pepper_var
        if "pepper_kalman_gain" not in data:
            data["pepper_kalman_gain"] = {}
        data["pepper_kalman_gain"]["INTARIAN_PEPPER_ROOT"] = kalman_gain
        if "pepper_prev_kalman_gain" not in data:
            data["pepper_prev_kalman_gain"] = {}
        data["pepper_prev_kalman_gain"]["INTARIAN_PEPPER_ROOT"] = kalman_gain

        # Purge deprecated data structures
        for key in ["prev_mid", "ema_mid", "ema_ret", "ema_abs_ret"]:
            if key in data: data.pop(key, None)

        return orders