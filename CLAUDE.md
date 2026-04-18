# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# FinoVITa: Round 2 "QUANT ARSENAL"

## Project Overview

High-frequency market-making and statistical arbitrage bot for IMC Prosperity 4.

- **Team:** RAV3N
- **Phase:** Round 2 - "Growing Your Outpost"
- **Target PnL:** > 200,000 XIRECs (Combined Manual + Algo)

## Core Mechanical Constraints (DO NOT OVERRIDE)

- **Position Limits:** Hard limit of **80** for both assets.
- **Market Access Fee (MAF):** The `bid()` function must **strictly return 2141**.
- **Execution:** Server-Safe V13 logic (FIFO Queue Priority).

## IMC Prosperity 4: Core Simulation Context

### 1. The Environment: Intara
- **Tick Speed:** Each "day" of trading is 10,000 ticks (0 to 9,999).
- **Latency:** `Trader` class called once per tick. Orders sent at Tick N are processed, results visible at Tick N+1.

### 2. The Matching Engine: FIFO (Price-Time Priority)
- Orders must be at the top of the book to get filled.
- Same price = whoever sent first gets priority.
- **Strategy:** Use "Pennying" (Best Bid + 1 / Best Ask - 1) to jump to the front of the queue.

### 3. Product Profiles (Round 2)

| Product | Behavior | Fair Value | Objective |
|---------|----------|------------|-----------|
| ASH_COATED_OSMIUM | Stable, high-liquidity, mean-reverting | Anchored at 10,000 | Market-make around 10k, high turnover |
| INTARIAN_PEPPER_ROOT | High volatility, trending, directional | Dynamic (OBI + momentum + flow) | Statistical arbitrage, micro-price prediction |

### 4. Round 2 Mechanics
- **Blind Auction:** One-time fee at round start determines "Market Access" (25% extra volume/flow).
- **Manual Challenge:** 50k XIRECs budget across Research, Scale, Speed (handled by teammate).

### 5. Success Metrics
- **Net PnL:** Cumulative profit minus Market Access Fee.
- **Sharpe Ratio:** Prioritize consistent, low-drawdown growth.

## Technical Stack

- **Languages:** Python 3.x (No external libraries except NumPy/Pandas/jsonpickle).
- **Architecture:** `trader.py` (Main), `datamodel.py` (Core Models).
- **Math:** Exponential Stoikov Risk Aversion, Logarithmic Inventory Skew, Colonel Blotto Resource Allocation.

## Development Patterns

### 1. State Persistence
Always use `traderData` (JSON serialization) to pass history across ticks. Ensure `_serialize_state` and `_restore_state` are updated when adding new technical indicators.

### 2. Signal Tapering (V13 Implementation)
- As `abs(position)` approaches 80, `alpha` signal must be linearly decayed.
- **Taper Zone:** Start decay at 70 units.
- **Goal:** Prevent "Inventory Lock" where bot hits limit and cannot capture new alpha.

### 3. Error Handling
The IMC environment is brittle. Wrap all math functions (log, exp) in safety checks to prevent `OverflowError` or `ValueError` which would crash the bot.

## Repository Commands

- **Backtest:** `prosperity4btx trader.py 2`
- **Clean State:** `rm -rf __pycache__`
