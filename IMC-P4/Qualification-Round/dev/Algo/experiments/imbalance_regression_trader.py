from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import math
import json
import numpy as np
import pandas as pd

class Trader:
    """
    V11: The Institutional Alpha-Matrix
    Features:
    1. Deep Volume-Weighted Imbalance (NumPy Arrays)
    2. Rolling OLS Linear Regression (Self-Tuning Signal Weights)
    3. State Serialization (Bypassing Tick Amnesia)
    4. Avellaneda-Stoikov Dynamic Volatility Spreads
    """

    def __init__(self):
        self.POSITION_LIMITS = {
            "ASH_COATED_OSMIUM": 20,
            "INTARIAN_PEPPER_ROOT": 20
        }
        
        # Hyperparameters for the Regression Engine
        self.LOOKBACK_WINDOW = 40
        self.DEFAULT_BETA = 4.5 # Fallback if regression lacks data
        
        # State variables to persist across ticks
        self.pepper_mids = []
        self.pepper_imb = []

    def run(self, state: TradingState):
        result = {}
        
        # 1. RESURRECT STATE (Overcoming Tick Amnesia)
        self._deserialize_state(state.traderData)

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)
            buy_cap = limit - position
            sell_cap = -limit - position

            # ---------------------------------------------------------
            # ENGINE 1: OSMIUM (Fixed Skew Cash Cow)
            # ---------------------------------------------------------
            if product == "ASH_COATED_OSMIUM":
                fair_value = 10000
                skew = position * 0.2 
                
                bid_price = math.floor(fair_value - 2 - skew)
                ask_price = math.ceil(fair_value + 2 - skew)
                
                if buy_cap > 0: orders.append(Order(product, bid_price, buy_cap))
                if sell_cap < 0: orders.append(Order(product, ask_price, sell_cap))
                result[product] = orders

            # ---------------------------------------------------------
            # ENGINE 2: INTARIAN PEPPER ROOT (Live Regression)
            # ---------------------------------------------------------
            if product == "INTARIAN_PEPPER_ROOT":
                if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
                    
                    # A. Vectorized Order Book Extraction
                    # We extract all levels of the book, not just Level 1
                    bids = np.array(list(order_depth.buy_orders.items())) # [[price, vol], ...]
                    asks = np.array(list(order_depth.sell_orders.items()))
                    
                    best_bid = bids[:, 0].max()
                    best_ask = asks[:, 0].min()
                    mid_price = (best_ask + best_bid) / 2.0
                    
                    # B. Deep Volume-Weighted Imbalance (DOBI)
                    # Instead of treating all volume equally, volume further from the mid is discounted
                    bid_distances = mid_price - bids[:, 0]
                    ask_distances = asks[:, 0] - mid_price
                    
                    # Weight volume by inverse distance to mid (prevents spoofing far away in the book)
                    weighted_bid_vol = np.sum(bids[:, 1] / (bid_distances + 1))
                    weighted_ask_vol = np.sum(np.abs(asks[:, 1]) / (ask_distances + 1))
                    
                    total_vol = weighted_bid_vol + weighted_ask_vol
                    imbalance = (weighted_bid_vol - weighted_ask_vol) / total_vol if total_vol > 0 else 0
                    
                    # C. Update Memory Arrays
                    self.pepper_mids.append(mid_price)
                    self.pepper_imb.append(imbalance)
                    
                    if len(self.pepper_mids) > self.LOOKBACK_WINDOW:
                        self.pepper_mids.pop(0)
                        self.pepper_imb.pop(0)
                    
                    # D. Live OLS Regression (The Machine Learning Engine)
                    # We want to find Beta: How much does the price move for every 1.0 shift in imbalance?
                    beta = self.DEFAULT_BETA
                    volatility = 0.0
                    
                    if len(self.pepper_mids) >= 10:
                        # Convert to pandas series for fast diff and rolling metrics
                        mids_s = pd.Series(self.pepper_mids)
                        imb_s = pd.Series(self.pepper_imb)
                        
                        # Price changes
                        delta_p = mids_s.diff().fillna(0).values
                        imb_shifted = imb_s.shift(1).fillna(0).values # Previous tick's imbalance
                        
                        # Run OLS Regression: DeltaP = Beta * Imbalance
                        # Using numpy linear algebra (lstsq)
                        x = imb_shifted.reshape(-1, 1)
                        y = delta_p
                        
                        # Avoid singularity / zero-variance crashes
                        if np.var(x) > 1e-8:
                            # lstsq returns [weights, residuals, rank, singular_values]
                            beta_calc = np.linalg.lstsq(x, y, rcond=None)[0][0]
                            # Cap the beta so the bot doesn't hallucinate extreme weights during spikes
                            beta = np.clip(beta_calc, 0.0, 10.0) 
                            
                        # Calculate rolling variance for Stoikov Model
                        volatility = mids_s.std()
                    
                    # E. Predictive Micro-Price Calculation
                    # The bot adjusts its fair value mathematically based on the exact live regression weight
                    micro_price = mid_price + (imbalance * beta)
                    
                    # F. Stoikov Risk Management
                    risk_aversion = 0.06
                    reservation_price = micro_price - (position * risk_aversion * volatility)
                    
                    optimal_spread = min(2.5 + (volatility * 0.4), 6.0)
                    
                    # G. Final Quote Generation
                    bid_quote = math.floor(reservation_price - (optimal_spread / 2))
                    ask_quote = math.ceil(reservation_price + (optimal_spread / 2))
                    
                    # Guardrails
                    bid_quote = min(bid_quote, best_bid)
                    ask_quote = max(ask_quote, best_ask)
                    if bid_quote >= ask_quote:
                        bid_quote = ask_quote - 1
                        
                    if buy_cap > 0: orders.append(Order(product, bid_quote, buy_cap))
                    if sell_cap < 0: orders.append(Order(product, ask_quote, sell_cap))
                    
                result[product] = orders

        # 2. SAVE STATE (Injecting memory into the next tick)
        next_trader_data = self._serialize_state()
        
        return result, 0, next_trader_data

    # ---------------------------------------------------------
    # STATE MANAGEMENT UTILITIES
    # ---------------------------------------------------------
    def _serialize_state(self) -> str:
        """Compress arrays to JSON so they survive tick amnesia."""
        state_dict = {
            "pm": [float(x) for x in self.pepper_mids],
            "pi": [float(x) for x in self.pepper_imb]
        }
        return json.dumps(state_dict)

    def _deserialize_state(self, trader_data: str):
        """Rebuild numpy history from the previous tick's JSON."""
        if not trader_data:
            return
        try:
            state_dict = json.loads(trader_data)
            self.pepper_mids = state_dict.get("pm", [])
            self.pepper_imb = state_dict.get("pi", [])
        except Exception:
            pass # Failsafe: if JSON is corrupted, start fresh
        