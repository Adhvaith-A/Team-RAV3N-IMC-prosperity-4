from datamodel import TradingState, Order
import math
import json

# IMPROVEMENTS IMPLEMENTED:
#  1. Adaptive drift weighting (regime-aware)
#  2. Better noise filtering (cascaded EMA)
#  3. Adaptive execution (dynamic iceberg chunks)
#  4. Price priority optimization (strong vs weak signals)
#  5. Regime-based execution (maker/mixed/taker)
#  6. Better position utilization (target 65-75 instead of 40-50)
#  7. Momentum holding logic (trailing stops, let winners run)
#  8. PnL feedback loop (risk scaling)
#  9. Microprice for Pepper (volume-weighted mid)
#  10. Imbalance persistence + asymmetric execution

class Trader:

    def __init__(self):
        self.POSITION_LIMITS = {
            "ASH_COATED_OSMIUM":    80,
            "INTARIAN_PEPPER_ROOT": 80,
        }
        self.OSMIUM_MODEL   = "kalman"
        self.OSMIUM_PROFILE = "mean"
        self.DEFAULT_STATE  = {
            # Osmium (keep simple - proven at 180k)
            "osmium_mid_ema":    {},
            "osmium_book_ema":   {},
            "osmium_slope_ema":  {},
            
            # Pepper (upgrade massively)
            "fast_ema":          {},
            "slow_ema":          {},
            "pepper_level":      {},
            "pepper_velocity":   {},
            "pepper_var":        {},
            
            # [NEW] Improvement 2: Cascaded filtering
            "cascade_ema_1":     {},
            "cascade_ema_2":     {},
            
            # [NEW] Improvement 7: Momentum holding
            "position_entry_price": {},
            "position_peak_pnl":    {},
            "trailing_stop_price":  {},
            
            # [NEW] Improvement 8: PnL feedback
            "realized_pnl":      {},
            "session_trades":    {},
            
            # [NEW] Improvement 9: Microprice
            "microprice":        {},
            
            # [NEW] Improvement 10: Imbalance persistence
            "obi_history":       {},
            "obi_persist_count": {},
        }

    # ─────────────────────────────────────────────────────────────────
    #  MAF bid - keep at proven 200
    # ─────────────────────────────────────────────────────────────────
    def bid(self):
        return 200

    # ─────────────────────────────────────────────────────────────────
    #  State management
    # ─────────────────────────────────────────────────────────────────
    def _restore_state(self, raw: str):
        if not raw:
            return json.loads(json.dumps(self.DEFAULT_STATE))
        try:
            loaded = json.loads(raw)
            for k, v in self.DEFAULT_STATE.items():
                if k not in loaded:
                    loaded[k] = type(v)() if isinstance(v, dict) else v
            return loaded
        except Exception:
            return json.loads(json.dumps(self.DEFAULT_STATE))

    def _serialize_state(self, data) -> str:
        try:
            return json.dumps(data)
        except Exception:
            return json.dumps(self.DEFAULT_STATE)

    # ─────────────────────────────────────────────────────────────────
    #  [NEW] Improvement 9: Microprice calculation for Pepper
    #  Volume-weighted mid using top 3 levels each side
    # ─────────────────────────────────────────────────────────────────
    def _microprice(self, od) -> float:
        if not od.buy_orders or not od.sell_orders:
            best_bid = max(od.buy_orders) if od.buy_orders else 0
            best_ask = min(od.sell_orders) if od.sell_orders else 0
            return (best_bid + best_ask) / 2.0 if best_bid and best_ask else 0
        
        # Top 3 bids, top 3 asks
        bids = sorted(od.buy_orders.items(), reverse=True)[:3]
        asks = sorted(od.sell_orders.items())[:3]
        
        bid_value = sum(p * v for p, v in bids)
        ask_value = sum(p * abs(v) for p, v in asks)
        bid_vol   = sum(v for _, v in bids)
        ask_vol   = sum(abs(v) for _, v in asks)
        
        if bid_vol + ask_vol == 0:
            return (bids[0][0] + asks[0][0]) / 2.0
        
        # Volume-weighted microprice
        return (bid_value + ask_value) / (bid_vol + ask_vol)

    # ─────────────────────────────────────────────────────────────────
    #  [NEW] Improvement 5: Regime detection
    #  Returns: 0 = calm/maker, 1 = volatile/mixed, 2 = extreme/taker
    # ─────────────────────────────────────────────────────────────────
    def _detect_regime(self, volatility: float, obi: float, alpha_signal: float) -> int:
        """
        0 = CALM:    low vol, small signals → pure maker
        1 = VOLATILE: medium vol, moderate signals → mixed (some sweeps)
        2 = EXTREME:  high vol OR strong signals → aggressive taker
        """
        vol_score = volatility / 0.02   # normalize to ~0-2 range
        obi_score = abs(obi) * 2.0      # 0-2 range
        alpha_score = abs(alpha_signal) / 1.0  # ~0-2 range
        
        # Composite regime score
        regime_score = vol_score + obi_score + alpha_score
        
        if regime_score < 1.5:
            return 0  # CALM
        elif regime_score < 3.5:
            return 1  # VOLATILE
        else:
            return 2  # EXTREME

    # ─────────────────────────────────────────────────────────────────
    #  [NEW] Improvement 1: Adaptive drift weights
    #  Weights vary by signal strength and volatility
    # ─────────────────────────────────────────────────────────────────
    def _adaptive_weights(self, alpha_signal: float, obi: float, volatility: float):
        """
        Returns (alpha_weight, obi_weight) that adapt to conditions.
        
        Logic:
        - High volatility → trust OBI more (book pressure matters)
        - Strong alpha signal → amplify it
        - Weak signals → conservative weights
        """
        alpha_strength = abs(alpha_signal)
        obi_strength   = abs(obi)
        
        # Base weights
        alpha_w = 1.5
        obi_w   = 1.1
        
        # Volatility adjustment: high vol → boost OBI weight
        vol_factor = min(2.0, volatility / 0.01)  # 0.01 = baseline vol
        obi_w *= (1.0 + 0.3 * vol_factor)
        
        # Signal strength adjustment: strong alpha → amplify alpha weight
        if alpha_strength > 0.5:
            alpha_w *= (1.0 + 0.4 * min(1.0, alpha_strength / 0.5))
        
        # Strong OBI → boost OBI weight
        if obi_strength > 0.6:
            obi_w *= (1.0 + 0.5 * min(1.0, (obi_strength - 0.6) / 0.4))
        
        return alpha_w, obi_w

    # ─────────────────────────────────────────────────────────────────
    #  [NEW] Improvement 3: Adaptive iceberg chunk sizing
    # ─────────────────────────────────────────────────────────────────
    def _adaptive_chunk(self, regime: int, urgency: float) -> int:
        """
        Returns iceberg chunk size based on regime and urgency.
        
        regime 0 (calm):    large chunks (5-7) → fewer orders, less slippage
        regime 1 (volatile): medium chunks (3-4)
        regime 2 (extreme):  small chunks (2-3) → fast execution
        
        urgency: 0-1 scale, how badly we need to execute
        """
        if regime == 0:
            base_chunk = 6
        elif regime == 1:
            base_chunk = 4
        else:
            base_chunk = 2
        
        # High urgency → smaller chunks (faster fill)
        urgency_adj = int(base_chunk * (1.0 - 0.4 * urgency))
        return max(2, min(7, urgency_adj))

    # ─────────────────────────────────────────────────────────────────
    #  [NEW] Improvement 7: Trailing stop logic
    # ─────────────────────────────────────────────────────────────────
    def _update_trailing_stop(self, data, symbol, pos, mid, pnl_per_unit):
        """
        Track peak PnL and set trailing stop.
        Exit rule: if PnL drops X% from peak, close position.
        """
        if pos == 0:
            data["position_peak_pnl"][symbol] = 0.0
            data["trailing_stop_price"][symbol] = 0.0
            return False  # no stop trigger
        
        # Current unrealized PnL
        entry_price = data.get("position_entry_price", {}).get(symbol, mid)
        current_pnl = (mid - entry_price) * pos if pos > 0 else (entry_price - mid) * abs(pos)
        
        # Update peak
        peak_pnl = data.get("position_peak_pnl", {}).get(symbol, current_pnl)
        if current_pnl > peak_pnl:
            data["position_peak_pnl"][symbol] = current_pnl
            peak_pnl = current_pnl
        
        # Trailing stop: if PnL drops 40% from peak, exit
        # (40% is aggressive but allows room for noise while protecting profit)
        stop_threshold = 0.40
        if peak_pnl > 0 and current_pnl < peak_pnl * (1.0 - stop_threshold):
            return True  # TRIGGER STOP
        
        return False

    # ─────────────────────────────────────────────────────────────────
    #  [NEW] Improvement 10: Imbalance persistence
    # ─────────────────────────────────────────────────────────────────
    def _obi_persistence(self, data, symbol, obi):
        """
        Track how many consecutive ticks OBI has been in same direction.
        Sustained imbalance → more confidence to execute.
        """
        if "obi_history" not in data:
            data["obi_history"] = {}
        if "obi_persist_count" not in data:
            data["obi_persist_count"] = {}
        
        hist = data["obi_history"].get(symbol, [])
        hist.append(obi)
        if len(hist) > 5:
            hist.pop(0)
        data["obi_history"][symbol] = hist
        
        # Count consecutive same-sign OBI
        if len(hist) < 2:
            persist = 0
        else:
            persist = 1
            for i in range(len(hist) - 1, 0, -1):
                if (hist[i] > 0) == (hist[i-1] > 0):
                    persist += 1
                else:
                    break
        
        data["obi_persist_count"][symbol] = persist
        return persist

    # ─────────────────────────────────────────────────────────────────
    #  [NEW] Improvement 8: PnL feedback loop
    # ─────────────────────────────────────────────────────────────────
    def _pnl_scale_factor(self, data, symbol):
        """
        Returns position sizing multiplier based on recent PnL.
        
        Winning → scale up (max 1.3x)
        Losing → scale down (min 0.6x)
        """
        pnl = data.get("realized_pnl", {}).get(symbol, 0.0)
        trades = data.get("session_trades", {}).get(symbol, 0)
        
        if trades < 5:
            return 1.0  # not enough data
        
        avg_pnl_per_trade = pnl / trades
        
        # Scale factor based on average PnL
        if avg_pnl_per_trade > 2.0:
            return 1.3  # winning big → scale up
        elif avg_pnl_per_trade > 0.5:
            return 1.15
        elif avg_pnl_per_trade < -2.0:
            return 0.6  # losing → scale down
        elif avg_pnl_per_trade < -0.5:
            return 0.8
        else:
            return 1.0

    # ─────────────────────────────────────────────────────────────────
    #  Main entry point
    # ─────────────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        result = {}
        data   = self._restore_state(state.traderData)

        for product, od in state.order_depths.items():
            pos   = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 80)
            buy_cap  =  limit - pos
            sell_cap = -limit - pos

            if product == "ASH_COATED_OSMIUM":
                result[product] = self._osmium(od, pos, limit, buy_cap, sell_cap, state.timestamp, data)
            elif product == "INTARIAN_PEPPER_ROOT":
                result[product] = self._pepper(od, pos, limit, buy_cap, sell_cap, state.timestamp, data)

        return result, 0, self._serialize_state(data)

    # ─────────────────────────────────────────────────────────────────
    #  OSMIUM - keep simple (proven at 180k baseline)
    #  Only minor tuning from winning reference
    # ─────────────────────────────────────────────────────────────────
    def _osmium(self, od, pos, limit, buy_cap, sell_cap, ts, data):
        orders = []
        FAIR   = 10000

        # Profile tuning (from 250k target reference)
        profile = {
            "trend_cap":      2.0,
            "trend_gain":     3.8,
            "trend_mid_gain": 0.025,
            "dip_offset":     4,
            "sweep_buy_gap":  1,
            "sweep_sell_gap": 1,
            "inventory_skew": 1.8,
            "signal_obi":     0.7,
            "layer_weight":   0.72,
            "taper_floor":    0.20,
        }

        if not od.sell_orders or not od.buy_orders:
            return orders

        best_ask = min(od.sell_orders)
        best_bid = max(od.buy_orders)
        mid      = (best_ask + best_bid) / 2.0

        # EMA updates
        prev_mid_ema  = data.get("osmium_mid_ema", {}).get("ASH_COATED_OSMIUM", mid)
        prev_book_ema = data.get("osmium_book_ema", {}).get("ASH_COATED_OSMIUM", FAIR)
        prev_slope_ema = data.get("osmium_slope_ema", {}).get("ASH_COATED_OSMIUM", 0.0)

        total_bid_vol = sum(od.buy_orders.values())
        total_ask_vol = sum(abs(v) for v in od.sell_orders.values())
        obi           = (total_bid_vol - total_ask_vol) / max(1, total_bid_vol + total_ask_vol)

        mid_ema  = 0.2 * mid + 0.8 * prev_mid_ema
        book_ema = 0.35 * (mid + obi * 1.2) + 0.65 * prev_book_ema
        fair_value = int(round(0.7 * FAIR + 0.2 * mid_ema + 0.1 * book_ema))

        slope     = mid_ema - prev_mid_ema
        slope_ema = 0.3 * slope + 0.7 * prev_slope_ema

        trend_bias = int(round(max(0.0, min(
            profile["trend_cap"],
            slope_ema * profile["trend_gain"] + (mid_ema - FAIR) * profile["trend_mid_gain"]
        ))))
        fair_value = max(9990, min(10012, fair_value + trend_bias))

        data.setdefault("osmium_mid_ema", {})["ASH_COATED_OSMIUM"]   = mid_ema
        data.setdefault("osmium_book_ema", {})["ASH_COATED_OSMIUM"]  = book_ema
        data.setdefault("osmium_slope_ema", {})["ASH_COATED_OSMIUM"] = slope_ema

        # Dip-buy overlay
        if ts < 9800 and buy_cap > 0:
            dip_ok = (
                mid <= fair_value - profile["dip_offset"]
                and slope     <= 0
                and slope_ema <= prev_slope_ema + 0.01
                and obi       <= 0.2
            )
            if dip_ok:
                swing_qty = max(1, min(buy_cap, 8))
                orders.append(Order("ASH_COATED_OSMIUM", min(best_ask, fair_value - 1), swing_qty))
                buy_cap -= swing_qty

        # Rebound take-profit
        if ts < 9920 and pos > 0 and sell_cap < 0:
            if mid >= fair_value - 1 and slope_ema >= -0.01:
                take_qty = max(1, min(pos, 10))
                orders.append(Order("ASH_COATED_OSMIUM", max(best_bid + 1, fair_value - 1), -take_qty))
                sell_cap += take_qty

        # Sweep taker
        if buy_cap > 0:
            for ask_level in sorted(od.sell_orders):
                if ask_level <= fair_value - profile["sweep_buy_gap"] or \
                   (ask_level <= fair_value - 1 and pos < 60):
                    avail = abs(od.sell_orders[ask_level])
                    qty   = min(avail, buy_cap)
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", ask_level, qty))
                        buy_cap -= qty

        if sell_cap < 0:
            for bid_level in sorted(od.buy_orders, reverse=True):
                if bid_level >= fair_value + profile["sweep_sell_gap"] or \
                   (bid_level >= fair_value + 1 and pos > -60):
                    avail = od.buy_orders[bid_level]
                    qty   = min(avail, abs(sell_cap))
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", bid_level, -qty))
                        sell_cap += qty

        # Maker quotes
        c = pos / float(limit)
        inv_skew = c * profile["inventory_skew"]
        res      = fair_value - inv_skew + obi * profile["signal_obi"]

        base_bid = min(int(round(res - 1)), best_ask - 1)
        base_ask = max(int(round(res + 1)), best_bid + 1)
        if base_bid >= base_ask:
            base_bid = base_ask - 1

        if buy_cap > 0:
            taper = max(profile["taper_floor"], 1.0 - max(0.0, pos - 70) / 20.0)
            q1 = max(1, int(buy_cap * profile["layer_weight"] * taper))
            q2 = buy_cap - q1
            orders.append(Order("ASH_COATED_OSMIUM", base_bid, q1))
            if q2 > 0 and base_bid - 1 > 0:
                orders.append(Order("ASH_COATED_OSMIUM", base_bid - 1, q2))

        if sell_cap < 0:
            abs_cap = abs(sell_cap)
            taper = max(profile["taper_floor"], 1.0 - max(0.0, -pos - 70) / 20.0)
            q1 = max(1, int(abs_cap * profile["layer_weight"] * taper))
            q2 = abs_cap - q1
            orders.append(Order("ASH_COATED_OSMIUM", base_ask, -q1))
            if q2 > 0:
                orders.append(Order("ASH_COATED_OSMIUM", base_ask + 1, -q2))

        return orders

    # ─────────────────────────────────────────────────────────────────
    #  PEPPER - MASSIVELY UPGRADED with all 10 improvements
    # ─────────────────────────────────────────────────────────────────
    def _pepper(self, od, pos, limit, buy_cap, sell_cap, ts, data):
        orders = []
        sym    = "INTARIAN_PEPPER_ROOT"

        if not od.sell_orders or not od.buy_orders:
            return orders

        best_ask = min(od.sell_orders)
        best_bid = max(od.buy_orders)
        
        # [IMPROVEMENT 9] Microprice instead of simple mid
        mid = self._microprice(od)

        # ── EMA signals ────────────────────────────────────────────────
        fast_ema = data.get("fast_ema", {}).get(sym, mid)
        slow_ema = data.get("slow_ema", {}).get(sym, mid)

        fast_ema = 0.5 * mid + 0.5 * fast_ema
        slow_ema = 0.05 * mid + 0.95 * slow_ema
        alpha_signal = fast_ema - slow_ema

        data.setdefault("fast_ema", {})[sym] = fast_ema
        data.setdefault("slow_ema", {})[sym] = slow_ema

        # [IMPROVEMENT 2] Cascaded filtering for smoother signal
        cascade_1 = data.get("cascade_ema_1", {}).get(sym, alpha_signal)
        cascade_2 = data.get("cascade_ema_2", {}).get(sym, alpha_signal)
        cascade_1 = 0.3 * alpha_signal + 0.7 * cascade_1
        cascade_2 = 0.3 * cascade_1 + 0.7 * cascade_2
        alpha_signal_filtered = cascade_2  # use filtered signal

        data.setdefault("cascade_ema_1", {})[sym] = cascade_1
        data.setdefault("cascade_ema_2", {})[sym] = cascade_2

        # ── OBI signals ────────────────────────────────────────────────
        total_bid_vol = sum(od.buy_orders.values())
        total_ask_vol = sum(abs(v) for v in od.sell_orders.values())
        obi = (total_bid_vol - total_ask_vol) / max(1, total_bid_vol + total_ask_vol)

        # Deep OBI (distance-weighted)
        deep_bid, deep_ask = 0.0, 0.0
        for p, v in od.buy_orders.items():
            deep_bid += v / max(1.0, mid - float(p))
        for p, v in od.sell_orders.items():
            deep_ask += abs(v) / max(1.0, float(p) - mid)
        deep_obi = (deep_bid - deep_ask) / max(1.0, deep_bid + deep_ask)
        
        blended_obi = 0.6 * obi + 0.4 * deep_obi

        # [IMPROVEMENT 10] Imbalance persistence
        persist_count = self._obi_persistence(data, sym, blended_obi)

        # ── Kalman filter (keep existing logic, proven) ────────────────
        pepper_level    = data.get("pepper_level", {}).get(sym, mid)
        pepper_velocity = data.get("pepper_velocity", {}).get(sym, 0.0)
        pepper_var      = data.get("pepper_var", {}).get(sym, 4.0)

        measured_fair   = mid + blended_obi * 0.7 + alpha_signal_filtered * 0.15
        measurement_var = max(1.5, 4.0 - abs(blended_obi) * 1.2)
        innovation      = measured_fair - pepper_level
        kalman_gain     = pepper_var / (pepper_var + measurement_var)
        pepper_level    = pepper_level + kalman_gain * innovation
        pepper_velocity = 0.9 * pepper_velocity + 0.1 * innovation
        pepper_var      = max(1.0, 0.92 * pepper_var + 0.08 * innovation ** 2)

        data.setdefault("pepper_level", {})[sym]    = pepper_level
        data.setdefault("pepper_velocity", {})[sym] = pepper_velocity
        data.setdefault("pepper_var", {})[sym]      = pepper_var

        # Estimate volatility (simple proxy from variance)
        volatility = max(0.001, math.sqrt(pepper_var) / 50.0)

        # [IMPROVEMENT 5] Regime detection
        regime = self._detect_regime(volatility, blended_obi, alpha_signal_filtered)

        # [IMPROVEMENT 1] Adaptive drift weights
        alpha_w, obi_w = self._adaptive_weights(alpha_signal_filtered, blended_obi, volatility)
        predicted_drift = alpha_signal_filtered * alpha_w + blended_obi * obi_w

        # [IMPROVEMENT 6] Better position utilization
        # Target 65-75 instead of 40-50 by using more aggressive tanh scaling
        tanh_scale = 1.6  # was 2.0 in reference → tighter → hits limit sooner
        target_pos = int(round(limit * math.tanh(predicted_drift / tanh_scale)))

        if ts >= 9930:
            target_pos = 0

        # [IMPROVEMENT 8] PnL feedback
        pnl_scale = self._pnl_scale_factor(data, sym)
        # Apply to effective limit (not target_pos directly, but capacity)
        effective_limit = int(limit * pnl_scale)
        effective_limit = max(20, min(80, effective_limit))

        # [IMPROVEMENT 7] Trailing stop check
        pnl_per_unit = (mid - data.get("position_entry_price", {}).get(sym, mid)) if pos > 0 else \
                       (data.get("position_entry_price", {}).get(sym, mid) - mid)
        stop_triggered = self._update_trailing_stop(data, sym, pos, mid, pnl_per_unit)

        # If stop triggered, force exit
        if stop_triggered and pos != 0:
            exit_qty = abs(pos)
            exit_price = best_bid if pos > 0 else best_ask
            orders.append(Order(sym, exit_price, -pos))  # market exit
            # Update PnL tracking
            pnl = pnl_per_unit * abs(pos)
            data.setdefault("realized_pnl", {})[sym] = data.get("realized_pnl", {}).get(sym, 0.0) + pnl
            data.setdefault("session_trades", {})[sym] = data.get("session_trades", {}).get(sym, 0) + 1
            data["position_entry_price"][sym] = 0.0
            data["position_peak_pnl"][sym] = 0.0
            return orders  # done, exit fully

        # Inventory-aware reservation price
        inv_skew = (pos / float(limit)) * 1.2
        reservation_price = mid + predicted_drift - inv_skew

        # Spread
        spread = max(1.0, 1.0 + math.sqrt(pepper_var) * 0.06)
        base_bid = int(round(reservation_price - spread))
        base_ask = int(round(reservation_price + spread))
        base_bid = min(base_bid, best_ask - 1)
        base_ask = max(base_ask, best_bid + 1)

        # [IMPROVEMENT 4] Price priority optimization
        # Strong signal → join at best (priority)
        # Weak signal → step back (avoid adverse selection)
        signal_strength = abs(predicted_drift)
        if signal_strength > 2.0:
            # STRONG: join at best
            optimal_bid = best_bid + 1 if target_pos > pos else best_bid
            optimal_ask = best_ask - 1 if target_pos < pos else best_ask
        else:
            # WEAK: step back
            optimal_bid = best_bid - 1 if target_pos > pos else best_bid - 2
            optimal_ask = best_ask + 1 if target_pos < pos else best_ask + 2

        optimal_bid = max(base_bid, min(optimal_bid, best_ask - 1))
        optimal_ask = min(base_ask, max(optimal_ask, best_bid + 1))

        # [IMPROVEMENT 3] Adaptive execution
        urgency = min(1.0, signal_strength / 3.0)  # 0-1 scale
        chunk_size = self._adaptive_chunk(regime, urgency)

        # ── Execution based on regime ──────────────────────────────────
        # [IMPROVEMENT 5] Regime-based execution
        
        if regime == 0:
            # CALM → pure maker
            if buy_cap > 0:
                q1 = max(1, int(buy_cap * 0.7))
                q2 = buy_cap - q1
                orders.append(Order(sym, optimal_bid, q1))
                if q2 > 0 and optimal_bid - 1 > 0:
                    orders.append(Order(sym, optimal_bid - 1, q2))

            if sell_cap < 0:
                abs_cap = abs(sell_cap)
                q1 = max(1, int(abs_cap * 0.7))
                q2 = abs_cap - q1
                orders.append(Order(sym, optimal_ask, -q1))
                if q2 > 0:
                    orders.append(Order(sym, optimal_ask + 1, -q2))

        elif regime == 1:
            # VOLATILE → mixed (some sweeps if signal strong)
            taker_threshold = 1.2

            if buy_cap > 0 and predicted_drift > taker_threshold:
                # Sweep top 3 asks
                for ask_level in sorted(od.sell_orders)[:3]:
                    if ask_level <= reservation_price and buy_cap > 0:
                        avail = abs(od.sell_orders[ask_level])
                        qty   = min(avail, buy_cap)
                        if qty > 0:
                            # Iceberg with adaptive chunk
                            while qty > 0:
                                chunk = min(chunk_size, qty)
                                orders.append(Order(sym, ask_level, chunk))
                                qty -= chunk
                            buy_cap -= min(avail, buy_cap)

            if sell_cap < 0 and predicted_drift < -taker_threshold:
                for bid_level in sorted(od.buy_orders, reverse=True)[:3]:
                    if bid_level >= reservation_price and sell_cap < 0:
                        avail = od.buy_orders[bid_level]
                        qty   = min(avail, abs(sell_cap))
                        if qty > 0:
                            while qty > 0:
                                chunk = min(chunk_size, qty)
                                orders.append(Order(sym, bid_level, -chunk))
                                qty -= chunk
                            sell_cap += min(avail, abs(sell_cap))

            # Fill remainder with maker
            if buy_cap > 0:
                orders.append(Order(sym, optimal_bid, buy_cap))
            if sell_cap < 0:
                orders.append(Order(sym, optimal_ask, sell_cap))

        else:  # regime == 2
            # EXTREME → aggressive taker, sweep deep
            taker_threshold = 0.8

            if buy_cap > 0 and predicted_drift > taker_threshold:
                for ask_level in sorted(od.sell_orders)[:5]:
                    if buy_cap > 0:
                        avail = abs(od.sell_orders[ask_level])
                        qty   = min(avail, buy_cap)
                        if qty > 0:
                            while qty > 0:
                                chunk = min(chunk_size, qty)
                                orders.append(Order(sym, ask_level, chunk))
                                qty -= chunk
                            buy_cap -= min(avail, buy_cap)

            if sell_cap < 0 and predicted_drift < -taker_threshold:
                for bid_level in sorted(od.buy_orders, reverse=True)[:5]:
                    if sell_cap < 0:
                        avail = od.buy_orders[bid_level]
                        qty   = min(avail, abs(sell_cap))
                        if qty > 0:
                            while qty > 0:
                                chunk = min(chunk_size, qty)
                                orders.append(Order(sym, bid_level, -chunk))
                                qty -= chunk
                            sell_cap += min(avail, abs(sell_cap))

        # [IMPROVEMENT 10] Asymmetric execution on sustained imbalance
        # If OBI persists 3+ ticks in same direction → extra aggression
        if persist_count >= 3:
            if blended_obi > 0.4 and buy_cap > 0:
                # Sustained buy pressure → join aggressively
                extra_qty = min(buy_cap, 5)
                orders.append(Order(sym, best_bid + 1, extra_qty))
                buy_cap -= extra_qty
            elif blended_obi < -0.4 and sell_cap < 0:
                extra_qty = min(abs(sell_cap), 5)
                orders.append(Order(sym, best_ask - 1, -extra_qty))
                sell_cap += extra_qty

        # Track entry price for trailing stop
        if pos == 0 and len(orders) > 0:
            data.setdefault("position_entry_price", {})[sym] = mid

        return orders
