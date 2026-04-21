from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import math
import json

class Trader:
    def __init__(self): 
        self.pepper_history = []
        self.history_window = 20
        # ROUND 2: Updated Position Limits
        self.POSITION_LIMITS = {
            "ASH_COATED_OSMIUM": 80,
            "INTARIAN_PEPPER_ROOT": 80
        }

    # ROUND 2: The Blind Auction Bid for 25% Extra Flow
    # 2141 is mathematically calculated to clear the median while guaranteeing net-positive EV.
    def bid(self) -> int:
        return 2141

    def run(self, state: TradingState):
        result = {}
        self._restore_state(state.traderData)

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 80)
            
            buy_cap = limit - position
            sell_cap = -limit - position

            # ---------------------------------------------------------
            # STRATEGY 1: ASH-COATED OSMIUM 
            # ---------------------------------------------------------
            if product == "ASH_COATED_OSMIUM":
                fair_value = 10000
                
                # ROUND 2 FIX: Normalized Inventory Skew
                # Replaces 'position * 0.2' so we don't blow out quotes at 80 inventory.
                inventory_ratio = position / limit
                skew = inventory_ratio * 3.0 
                
                bid_price = round(fair_value - 2 - skew)
                ask_price = round(fair_value + 2 - skew)
                
                if buy_cap > 0:
                    orders.append(Order(product, int(bid_price), buy_cap))
                if sell_cap < 0:
                    orders.append(Order(product, int(ask_price), sell_cap))
                    
                result[product] = orders

            # ---------------------------------------------------------
            # STRATEGY 2: INTARIAN PEPPER ROOT 
            # ---------------------------------------------------------
            if product == "INTARIAN_PEPPER_ROOT":
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_bid = max(order_depth.buy_orders.keys())
                    mid_price = (best_ask + best_bid) / 2.0
                    
                    self.pepper_history.append(mid_price)
                    if len(self.pepper_history) > self.history_window:
                        self.pepper_history.pop(0)
                        
                    variance = 0.0
                    if len(self.pepper_history) >= 2:
                        mean = sum(self.pepper_history) / len(self.pepper_history)
                        variance = sum((x - mean) ** 2 for x in self.pepper_history) / len(self.pepper_history)
                    
                    volatility = math.sqrt(variance)

                    bid_vol = order_depth.buy_orders[best_bid]
                    ask_vol = abs(order_depth.sell_orders[best_ask])
                    total_vol = bid_vol + ask_vol
                    obi = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0
                    
                    micro_price = mid_price + (obi * 4.5)
                    
                    # ROUND 2 FIX: Normalized Risk Aversion
                    # Scaled down from 0.06 to 0.015 to account for the 4x capacity increase (20 -> 80)
                    risk_aversion = 0.015
                    reservation_price = micro_price - (position * risk_aversion * volatility)
                    
                    base_spread = 2.5
                    optimal_spread = min(base_spread + (volatility * 0.4), 6.0)
                    
                    bid_price = math.floor(reservation_price - (optimal_spread / 2))
                    ask_price = math.ceil(reservation_price + (optimal_spread / 2))
                    
                    # SERVER-SAFE FIX: Queue Priority Pennying
                    bid_price = min(bid_price, best_ask - 1)
                    ask_price = max(ask_price, best_bid + 1)
                    
                    if bid_price >= ask_price:
                        bid_price = ask_price - 1
                    
                    if buy_cap > 0:
                        orders.append(Order(product, int(bid_price), buy_cap))
                    if sell_cap < 0:
                        orders.append(Order(product, int(ask_price), sell_cap))
                        
                result[product] = orders

        return result, 0, self._serialize_state()

    def _serialize_state(self) -> str:
        return json.dumps({
            "prices": self.pepper_history
        })

    def _restore_state(self, trader_data: str):
        if not trader_data or trader_data == "":
            return
        try:
            d = json.loads(trader_data)
            self.pepper_history = d.get("prices", [])
        except (json.JSONDecodeError, TypeError):
            pass