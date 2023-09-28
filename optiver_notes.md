# Optivers - Trading at the Close

Final ten minutes of the trading day, often characterised by heightened volatility and rapid price fluctuations.
In the last ten minitues of Nasdaq exchange trading session, market makers like Optiver merge traditional order book data with auction book data.

**Goal**:
predict the stock closing price movements for hundreds of Nasdaq listed stocks
using data from the order book and the closing auction of the stock

Information from the auction can be used to adjust prices, assess supply and demand dynamics, and identify trading opportunities

## Dataset

Historic data for the daily ten nimute closing auction on NASDAQ stock exchange:

stock_id, data_id, imbalance_size, imbalance_buy_sell_flag, reference_price, matched_size, far_price, near_price, [bid/ask]_price, [bid/ask]_size, wap, seconds_in_bucket

continuous phase: best bid, best ask, best volume\
auction phase: reference price, matched volume, imbalnce

On-close orders (MOC, LOC, IO) together with orders from the continuous phase contribute to auction volume and imbalance.
The final price is determined to maximize the matched volume. Therefore, the volume to buy and to sell moves the final price.

## Background Knowledge

*Auction*: a mechanism for determing the price of a particular asset by allowing multiple buyers and sellers to interact directly in a controlled, regulated environmnet.

In a *closing auction*, orders are collected over a pre-determined timeframe and then matched at a single price determined by the buy & sell demand expressed by auction participants.

- [Optiver Realized Volatility Prediction explained with intro to financial concepts](https://www.kaggle.com/code/jiashenliu/introduction-to-financial-concepts-and-data)

### Order book statistics

relect market liquidity and stock valuation

**bid_ask_spread** = best_offer / best_bid - 1

**Weighted averaged price**, WAP = (bid_price_1 * ask_size_1 + ask_price_1 * bid_size_1) / (bid_size_1 + ask_size_1)

**log return**: how can we compare the price of a stock between yesterday and today?

$r_{t1, t2} = log({\frac{S_t2}{S_t1}})$, where $S_t$ is the price of stock $S$ at time $t$

- additive across time $r_{t1, t2} + r_{t2, t3} = r_{t1, t3}$
- log returns not bounded (regular returns cannot go below -100%)

**realized volatility**: standard deviation of the stock log returns, usually normalized to a 1-year period and annualized standard deviation is called volatility.
$$\sigma = \sqrt{{\Sigma}_{t}r^{2}_{t-1, t}}$$

## Plan

### Feature Engineering

stock_id, data_id, imbalance_size, imbalance_buy_sell_flag, reference_price, matched_size, far_price, near_price, [bid/ask]_price, [bid/ask]_size, wap, seconds_in_bucket

Create/transform features that might influence stock price movements.

Common features to consider:

- price change: diff between consecutive closing price
- moving averages: e.g. 10-day, 50-day averages of closing prices
- technical indicators: compute RSI (relative strength index), MACD (moving price convergence divergence) Bollinger Bands
- lagged features: past values often impact future prices
- volatility measures: e.g. historical volatility, implied volatility

### Data Preprocessing

Normalize if using linear regression etc.

**Time based splitting**:
when splitting training and testing datasets, make sure to keep time sequence intact.

### Model Selection

- LightGBM
- XGBoost
- LSTM (RNN)

### Evaluation

Metrics?
