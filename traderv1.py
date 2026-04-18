from datamodel import TradingState, Order
import math
import json

class Trader:
    """
    FinoVITa RAV3N - V26 "FULL LONG RIDER"
    ======================================
    We're breakeven but not capturing the 1000-pt trend.
    Need to: 1) Get full 80 units early, 2) Hold through day
    """

    def __init__(self):
        self.POSITION_LIMITS = {
            "ASH_COATED_OSMIUM": 80,
            "INTARIAN_PEPPER_ROOT": 80
        }
        self.DEFAULT_STATE = {
            "prev_mid": {},
            "ema_mid": {},
            "ema_ret": {},
            "ema_abs_ret": {},
            "osmium_mid_ema": {},
            "osmium_book_ema": {}
        }

    def run(self, state: TradingState):
        result = {}
        data = self._restore_state(state.traderData)

        for product in state.order_depths:
            od = state.order_depths[product]
            pos = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 80)
            buy_cap = limit - pos
            sell_cap = -limit - pos

            if product == "ASH_COATED_OSMIUM":
                result[product] = self._osmium(od, pos, limit, buy_cap, sell_cap, state.timestamp, data)
            elif product == "INTARIAN_PEPPER_ROOT":
                result[product] = self._pepper(od, pos, limit, buy_cap, sell_cap, state.timestamp, data)

        return result, 0, self._serialize_state(data)

    def _restore_state(self, trader_data: str):
        if not trader_data:
            return json.loads(json.dumps(self.DEFAULT_STATE))
        try:
            loaded = json.loads(trader_data)
            for key in self.DEFAULT_STATE:
                if key not in loaded or not isinstance(loaded[key], dict):
                    loaded[key] = {}
            return loaded
        except Exception:
            return json.loads(json.dumps(self.DEFAULT_STATE))

    def _serialize_state(self, data) -> str:
        try:
            return json.dumps(data)
        except Exception:
            return json.dumps(self.DEFAULT_STATE)

    def _osmium(self, od, pos, limit, buy_cap, sell_cap, timestamp, data):
        orders = []
        FAIR = 10000

        if not od.sell_orders or not od.buy_orders:
            return orders

        best_ask = min(od.sell_orders.keys())
        best_bid = max(od.buy_orders.keys())
        mid = (best_ask + best_bid) / 2.0

        if "osmium_mid_ema" not in data:
            data["osmium_mid_ema"] = {}
        if "osmium_book_ema" not in data:
            data["osmium_book_ema"] = {}

        prev_mid_ema = data["osmium_mid_ema"].get("ASH_COATED_OSMIUM", mid)
        prev_book_ema = data["osmium_book_ema"].get("ASH_COATED_OSMIUM", FAIR)

        total_bid_vol = sum(od.buy_orders.values())
        total_ask_vol = sum(abs(v) for v in od.sell_orders.values())
        total_vol = max(1, total_bid_vol + total_ask_vol)
        obi = (total_bid_vol - total_ask_vol) / total_vol

        mid_ema = 0.2 * mid + 0.8 * prev_mid_ema
        book_ema = 0.35 * (mid + obi * 1.2) + 0.65 * prev_book_ema
        fair_value = int(round(0.7 * FAIR + 0.2 * mid_ema + 0.1 * book_ema))
        fair_value = max(9990, min(10010, fair_value))
        slope = mid_ema - prev_mid_ema
        slope_ema_key = "osmium_slope_ema"
        if slope_ema_key not in data:
            data[slope_ema_key] = {}
        prev_slope_ema = data[slope_ema_key].get("ASH_COATED_OSMIUM", 0.0)
        slope_ema = 0.3 * slope + 0.7 * prev_slope_ema

        data["osmium_mid_ema"]["ASH_COATED_OSMIUM"] = mid_ema
        data["osmium_book_ema"]["ASH_COATED_OSMIUM"] = book_ema
        data[slope_ema_key]["ASH_COATED_OSMIUM"] = slope_ema

        # Dip-buy / rebound-sell overlay: buy only on a real dip, then take profit when price mean-reverts.
        if timestamp < 9700 and buy_cap > 0:
            dip_signal = mid <= fair_value - 3 and slope < 0 and slope_ema <= prev_slope_ema and obi <= 0.2
            if dip_signal:
                swing_qty = max(1, min(buy_cap, 6))
                swing_price = min(best_ask, fair_value - 1)
                orders.append(Order("ASH_COATED_OSMIUM", swing_price, swing_qty))
                buy_cap -= swing_qty

        if timestamp < 9900 and pos > 0 and sell_cap < 0:
            rebound_signal = mid >= fair_value - 1
            if rebound_signal:
                take_profit_qty = max(1, min(pos, 6))
                take_profit_price = max(best_bid + 1, fair_value - 1)
                orders.append(Order("ASH_COATED_OSMIUM", take_profit_price, -take_profit_qty))
                sell_cap += take_profit_qty

        # 1. Unbounded Sweep-Taker
        # Aggressively cross the spread when the book is far from the adaptive fair.
        if buy_cap > 0:
            for ask_level in sorted(od.sell_orders.keys()):
                if ask_level <= fair_value - 2 or (ask_level <= fair_value - 1 and pos < 60):
                    if buy_cap > 0:
                        avail = abs(od.sell_orders[ask_level])
                        qty = min(avail, buy_cap)
                        if qty > 0:
                            orders.append(Order("ASH_COATED_OSMIUM", ask_level, qty))
                            buy_cap -= qty

        if sell_cap < 0:
            for bid_level in sorted(od.buy_orders.keys(), reverse=True):
                if bid_level >= fair_value + 2 or (bid_level >= fair_value + 1 and pos > -60):
                    if sell_cap < 0:
                        avail = od.buy_orders[bid_level]
                        qty = min(avail, abs(sell_cap))
                        if qty > 0:
                            orders.append(Order("ASH_COATED_OSMIUM", bid_level, -qty))
                            sell_cap += qty

        # 2. Adaptive fair value with inventory-aware skew.
        c = pos / float(limit)
        inventory_skew = c * 2.0
        reservation_price = fair_value - inventory_skew

        # Keep the maker quotes close to the adaptive center without becoming sticky at the limit.
        signal = reservation_price + (obi * 0.8)

        base_bid = min(int(round(signal - 1)), best_ask - 1)
        base_ask = max(int(round(signal + 1)), best_bid + 1)
        if base_bid >= base_ask:
            base_bid = min(base_bid, base_ask - 1)

        # 3. Multi-Layer Volume Laddering with a gentle inventory taper.
        if buy_cap > 0:
            taper = max(0.25, 1.0 - max(0.0, pos - 70) / 20.0)
            qty_layer1 = max(1, int(buy_cap * (0.65 * taper)))
            qty_layer2 = buy_cap - qty_layer1
            orders.append(Order("ASH_COATED_OSMIUM", base_bid, qty_layer1))
            if qty_layer2 > 0 and base_bid - 1 > 0:
                orders.append(Order("ASH_COATED_OSMIUM", base_bid - 1, qty_layer2))

        if sell_cap < 0:
            abs_sell_cap = abs(sell_cap)
            taper = max(0.25, 1.0 - max(0.0, -pos - 70) / 20.0)
            qty_layer1 = max(1, int(abs_sell_cap * (0.65 * taper)))
            qty_layer2 = abs_sell_cap - qty_layer1
            orders.append(Order("ASH_COATED_OSMIUM", base_ask, -qty_layer1))
            if qty_layer2 > 0:
                orders.append(Order("ASH_COATED_OSMIUM", base_ask + 1, -qty_layer2))

        return orders

    def _pepper(self, od, pos, limit, buy_cap, sell_cap, timestamp, data):
        orders = []

        if not od.sell_orders or not od.buy_orders:
            return orders

        best_ask = min(od.sell_orders.keys())
        best_bid = max(od.buy_orders.keys())
        mid = (best_ask + best_bid) / 2.0

        if "fast_ema" not in data:
            data["fast_ema"] = {}
        if "slow_ema" not in data:
            data["slow_ema"] = {}

        fast_ema = data["fast_ema"].get("INTARIAN_PEPPER_ROOT", mid)
        slow_ema = data["slow_ema"].get("INTARIAN_PEPPER_ROOT", mid)

        # 1. Signal Generation: Fast EMA vs Slow EMA (Velocity)
        fast_ema = 0.5 * mid + 0.5 * fast_ema
        slow_ema = 0.05 * mid + 0.95 * slow_ema
        alpha_signal = fast_ema - slow_ema

        # 2. Cumulative Order Book Imbalance (Acceleration)
        total_bid_vol = sum(od.buy_orders.values())
        total_ask_vol = sum(abs(v) for v in od.sell_orders.values())
        total_vol = max(1, total_bid_vol + total_ask_vol)
        obi = (total_bid_vol - total_ask_vol) / total_vol

        # 3. Micro-Price Drift & Target Position
        # Combine trend velocity with momentary book acceleration
        predicted_drift = (alpha_signal * 1.5) + (obi * 1.0)
        
        # Continuous target positional bias using tanh limit
        target_pos = int(round(limit * math.tanh(predicted_drift / 2.0)))
        if timestamp >= 9850:
            target_pos = 0 # Unwind into the close safely

        # Calculate safety-anchored fair value incorporating inventory skew
        inventory_skew = (pos / float(limit)) * 1.5
        reservation_price = mid + predicted_drift - inventory_skew

        # Baseline maker quoting distances
        base_bid = int(round(reservation_price - 1.0))
        base_ask = int(round(reservation_price + 1.0))
        # Hard limits to never cross the spread passively
        base_bid = min(base_bid, best_ask - 1)
        base_ask = max(base_ask, best_bid + 1)

        # 4. Asymmetric Taker Protocol
        # Only cross the spread aggressively if the trend momentum completely overtakes the spread cost
        taker_threshold = 1.5
        
        if buy_cap > 0 and predicted_drift > taker_threshold:
            for ask_level in sorted(od.sell_orders.keys()):
                # Only take if we are buying cheaper than our true anticipated future price
                if ask_level <= reservation_price:
                    avail = abs(od.sell_orders[ask_level])
                    qty = min(avail, buy_cap)
                    if qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", ask_level, qty))
                        buy_cap -= qty

        if sell_cap < 0 and predicted_drift < -taker_threshold:
            for bid_level in sorted(od.buy_orders.keys(), reverse=True):
                # Only short if the bid is richer than our true anticipated future price
                if bid_level >= reservation_price:
                    avail = od.buy_orders[bid_level]
                    qty = min(avail, abs(sell_cap))
                    if qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", bid_level, -qty))
                        sell_cap += qty

        # 5. Inventory-Aware Maker Replenishment (No Blind Pennying)
        if buy_cap > 0:
            if target_pos > pos:
                # We actively want to increase position. Quote up to base_bid, which includes drift.
                # In strong uptrends, this inherently pennies safely.
                optimal_bid = min(best_bid + 1, base_bid)
            else:
                # We do not want to increase position. Step back and fade the market.
                optimal_bid = min(best_bid - 1, base_bid)
                
            qty_layer1 = max(1, int(buy_cap * 0.7))
            qty_layer2 = buy_cap - qty_layer1
            orders.append(Order("INTARIAN_PEPPER_ROOT", optimal_bid, qty_layer1))
            if qty_layer2 > 0 and optimal_bid - 1 > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", optimal_bid - 1, qty_layer2))

        if sell_cap < 0:
            abs_sell_cap = abs(sell_cap)
            if target_pos < pos:
                # We actively want to decrease position / short
                optimal_ask = max(best_ask - 1, base_ask)
            else:
                # We want to hold longs, fade the ask
                optimal_ask = max(best_ask + 1, base_ask)

            qty_layer1 = max(1, int(abs_sell_cap * 0.7))
            qty_layer2 = abs_sell_cap - qty_layer1
            orders.append(Order("INTARIAN_PEPPER_ROOT", optimal_ask, -qty_layer1))
            if qty_layer2 > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", optimal_ask + 1, -qty_layer2))

        # 6. Save Persistent State
        data["fast_ema"]["INTARIAN_PEPPER_ROOT"] = fast_ema
        data["slow_ema"]["INTARIAN_PEPPER_ROOT"] = slow_ema

        # Purge deprecated data structures
        for key in ["prev_mid", "ema_mid", "ema_ret", "ema_abs_ret"]:
            if key in data: data.pop(key, None)

        return orders
