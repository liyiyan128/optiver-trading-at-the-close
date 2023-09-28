# Optivers - Trading at the Close

Final ten minutes of the trading day, often characterised by heightened volatility and rapid price fluctuations.
In the last ten minitues of Nasdaq exchange trading session, market makers like Optiver merge traditional order book data with auction book data.

**Goal**:
predict the stock closing price movements for hundreds of Nasdaq listed stocks
using data from the order book and the closing auction of the stock

Information from the auction can be used to adjust prices, assess supply and demand dynamics, and identify trading opportunities

- [Optiver Trading At the Close intro](https://www.kaggle.com/code/tomforbes/optiver-trading-at-the-close-introduction)

## Dataset

Historic data for the daily ten nimute closing auction on NASDAQ stock exchange:

stock_id, data_id, imbalance_size, imbalance_buy_sell_flag, reference_price, matched_size, far_price, near_price, [bid/ask]_price, [bid/ask]_size, wap, seconds_in_bucket

continuous phase: best bid, best ask, best volume\
auction phase: reference price, matched volume, imbalnce

On-close orders (MOC, LOC, IO) together with orders from the continuous phase contribute to auction volume and imbalance.
The final price is determined to maximize the matched volume. Therefore, the volume to buy and to sell moves the final price.

## Background Knowledge

Previous Optiver Kaggle Chanllenge: [Optiver Realized Volatility Prediction explained with intro to financial concepts](https://www.kaggle.com/code/jiashenliu/introduction-to-financial-concepts-and-data)

*Auction*: a mechanism for determing the price of a particular asset by allowing multiple buyers and sellers to interact directly in a controlled, regulated environmnet.

In a *closing auction*, orders are collected over a pre-determined timeframe and then matched at a single price determined by the buy & sell demand expressed by auction participants.

### Auction order book

| Bid | Price | Ask |
| --- | ----- | --- |
|     | 10    | 1   |
| 3   | 9     | 2   |
| 4   | 8     | 4   |

Unlike order book for continuous trading, orders are not immediately matched, but instead collected until the moment the auction ends.

In the example above that unlike the order book for continuous trading, the highest bid is not greater than the lowest ask. The book in the example above is referred to as *in cross*, since the best bid and ask are overlapping. The closing auction price is therefore referred to as the *uncross price*, the price at which the shares which were in cross are matched.

- At a price of 10, 0 lots would be matched since there as no bids >= 10.
- At a price of 9, 3 lots would be matched, as there are 3 bids >= 9 and 6 asks <= 9.
- At a price of 8, 4 lots would be matched, since are 7 bids >= 8, and there are 4 asks <=8.

So the price which maximises the number of matched lots would be 8.

We would therefore describe the auction order book in the following way:

- The *uncross price* is 8.
- The *matched size* would be 4.
- The *imbalance* would be 3 lots in the buy direction.

The term *imbalance* refers to the number of unmatched shares.

The term *far price* refers to the hypothetical uncross price of the auction book, if it were to uncross at the reporting time.
Nasdaq provides far price information 5 minutes before the closing cross.

### Combined book

Combining the two books can result in a more accurate reflection of the market's buying and selling interest at different price levels.

Combine by aggregating the buying & selling interst across all price levels. The hypothetical uncross price of combined book is called the *near price*. Nasdaq provides near price information 5 minutes before the closing cross.

Nasdaq also provides an indication of the fair price called the *reference price*. The reference price is calculated as follows:

- If the near price is between the best bid and ask, then the reference price is equal to the near price
- If the near price > best ask, then reference price = best ask
- If the near price < best bid, then reference price = best bid So the reference price is the near price bounded between the best bid and ask.

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
