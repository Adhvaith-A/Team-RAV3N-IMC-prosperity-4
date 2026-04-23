# Team RAV3N Trading Bot

This repository contains the current codebase for the IMC Prosperity 4 trading system by team RAV3N.

# Team Members
## Adhvaith Anand
## Unnabh Majumdhar

The strategy is built around two distinct behaviors:

- `ASH_COATED_OSMIUM` is handled as a mean-reverting, inventory-aware market-making product.
- `INTARIAN_PEPPER_ROOT` is handled as a directional, orderflow-driven product with regime awareness.

## What Is In The Repo

- `trader.py` is the active bot.
- `datamodel.py` provides the Prosperity trading types.
- `CODE_DOCUMENTATION.md` explains the code from top to bottom.
- `Strategy.md` explains the thinking process behind the current design.
- `LICENSE` describes the viewing-only restrictions for this repository snapshot.

## Current Model Results

These are the current reference results for the deployed strategy snapshot:

- Rust backtester total: `255920`
- Prosperity4btx total: `256156`

Those numbers reflect the current tuned version of `trader.py` after the professor-style signal decomposition and the later optimization passes.

## Strategy Summary

The bot uses a layered approach:

- short-horizon orderflow and trend for Pepper
- MA-distance and cumulative imbalance for Osmium
- wavelet-based chop detection for regime handling
- dynamic Kalman-style adaptation for noisy conditions
- persistent JSON state across ticks
- volatility and correlation controls to avoid overexposure

## Important Constraints

- Position limit: `80` per product.
- `bid()` must return `2141` or a positive value, '2141' is specifically meant for a 25% bid.
- The code is tuned for the round-2 Prosperity environment and is not a general-purpose trading framework.

## How To Run

If you are validating locally, use the project’s existing backtest flow. The main entry points in this workspace have been:

- the Rust backtester in `prosperity_rust_backtester/`
- the Prosperity wrapper used earlier in the session

Typical validation is performed by running the bot through the backtester and checking the resulting PnL and drawdown behavior.

## Notes On Use

This repository is intentionally restricted for review only. Please read the license before using any part of the code.

## Project Goal

The goal of this bot is not to maximize aggression. The goal is to preserve the profitable parts of the signal stack while keeping execution selective enough to survive in a noisy FIFO market.
