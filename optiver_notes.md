# Optivers - Trading at the Close

- [Optivers - Trading at the Close](#optivers---trading-at-the-close)
  - [Dataset](#dataset)
    - [Overview](#overview)
  - [Background Knowledge](#background-knowledge)
    - [Auction order book](#auction-order-book)
    - [Combined book](#combined-book)
    - [Order book statistics](#order-book-statistics)
  - [Plan](#plan)
    - [Feature Engineering](#feature-engineering)
      - [Data Preprocessing](#data-preprocessing)
    - [Model Selection](#model-selection)
    - [Evaluation](#evaluation)

**TODO List**

- Feature engineering, cross-validation
- Make 10 - 20 submissions based on:
  - [lgb xgb catboost](https://www.kaggle.com/code/yuanzhezhou/baseline-lgb-xgb-and-catboost)
  - [feat eng + lgb](https://www.kaggle.com/code/renatoreggiani/optv-lightgbm)
  - [submission demo](https://www.kaggle.com/code/sohier/optiver-2023-basic-submission-demo)
- Model averaging

Final ten minutes of the trading day, often characterised by heightened volatility and rapid price fluctuations.
In the last ten minitues of Nasdaq exchange trading session, market makers like Optiver merge traditional order book data with auction book data.

**Goal**:
predict the stock closing price movements for hundreds of Nasdaq listed stocks
using data from the order book and the closing auction of the stock

Information from the auction can be used to adjust prices, assess supply and demand dynamics, and identify trading opportunities

Some useful material to look at:

- [Highlights of code and dicussion board](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/448920)

- [Weights of the synthetic index](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/442851) --> feature based on index?

- [Optiver Trading At the Close intro](https://www.kaggle.com/code/tomforbes/optiver-trading-at-the-close-introduction)
- [Extensive EDA](https://www.kaggle.com/code/ravi20076/optiver-extensiveeda)

Previous Optiver Kaggle [Optiver Realized Volatility Prediction explained with intro to financial concepts](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/overview):
- [1st place sol](https://www.kaggle.com/code/nyanpn/1st-place-public-2nd-place-solution)
- [LGBM baseline, feature engineering](https://www.kaggle.com/code/alexioslyon/lgbm-baseline)

Prerequisite:

- Gradient boosting (LightGBM, XGBoost)
  - [A gentle intro to gradient boosting](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)
- Neural networks (PyTorch, Keras)

- Regularisation (l1, l2)
- Cross-validation

## Dataset

Historic data for the daily ten nimute closing auction on NASDAQ stock exchange:

stock_id, data_id, imbalance_size, imbalance_buy_sell_flag, reference_price, matched_size, far_price, near_price, [bid/ask]_price, [bid/ask]_size, wap, seconds_in_bucket

continuous phase: best bid, best ask, best volume\
auction phase: reference price, matched volume, imbalnce

On-close orders (MOC, LOC, IO) together with orders from the continuous phase contribute to auction volume and imbalance.
The final price is determined to maximize the matched volume. Therefore, the volume to buy and to sell moves the final price.

### Overview

[Extensive EDA](https://www.kaggle.com/code/ravi20076/optiver-extensiveeda)

- 200 stocks, 481 trading days
- NaN in reference_price, matched_size, far_price, near_price, bid_price and wap\
  in 132 rows across stocks 19, 101, 131. These nulls are all located in the same location, i.e. across the same rows.\
  far_price, near_price have 55% NaN.
- Not all stocks are traded on all days and all time steps.\
  All sotcks end on date_id = 480 though some start late
- Outliers are present in several stocks.

- Several stocks are consistently underperforming with a downtrend.
- It is evident that quite a few stocks belong to the same sector based on the plots, and the overall stocks belong to multiple sectors colectively.

## Background Knowledge

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

**Bid ask spread**, $BidAskSpread = BestOffer / BestBid - 1$

**Weighted averaged price**
$$WAP = \frac{BidPrice * AskSize + AskPrice * BidSize}{BidSize + AskSize}$$

**log return**: how can we compare the price of a stock between yesterday and today?

$r_{t1, t2} = log({\frac{S_t2}{S_t1}})$, where $S_t$ is the price of stock $S$ at time $t$

- additive across time $r_{t1, t2} + r_{t2, t3} = r_{t1, t3}$
- log returns not bounded (regular returns cannot go below -100%)

**realized volatility**: standard deviation of the stock log returns, usually normalized to a 1-year period and annualized standard deviation is called volatility.
$$\sigma = \sqrt{{\Sigma}_{t}r^{2}_{t-1, t}}$$

## Plan

### Feature Engineering

`stock_id`, `data_id`, `imbalance_size`, `imbalance_buy_sell_flag`, `reference_price`, `matched_size`, `far_price`, `near_price`, `[bid/ask]_price`, `[bid/ask]_size`, `wap`, `seconds_in_bucket`

Create/transform features that might influence stock price movements.

1. Time-Based Features:

    Extract time-related information from the data_id or timestamp, such as day of the week, hour of the day, or minute of the hour. These features can capture potential patterns related to market hours, trading sessions, or daily fluctuations.

2. Moving Averages:

    Calculate various moving averages (e.g., 10-day, 50-day, 200-day) for the reference_price or other relevant price-related features. Moving averages can capture trends and smooth out short-term fluctuations.

3. Volatility Measures:

    Compute measures of price volatility, such as historical volatility or implied volatility, using features like wap or reference_price. Volatility can be an essential factor in stock price movements.

4. Technical Indicators:

    Calculate technical indicators commonly used in stock analysis, such as:
    Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Bollinger Bands, Stochastic Oscillator.\
    These indicators can help capture price momentum and overbought/oversold conditions.

5. Lagged Features:

    Create lagged versions of relevant features. For instance, you can use the closing price or volume from the previous time steps as input features, as past values often influence future prices.

6. Price Spread Features:

    Compute features related to the spread between bid and ask prices. This could include measures like bid-ask spread or bid-ask ratio, which provide insights into market liquidity.

7. Imbalance Ratio:

    You already have an imbalance_size feature. You can create derived features related to order imbalances, such as the ratio of buy orders to sell orders (imbalance_buy_sell_ratio).

8. Feature Interactions:

    Explore interactions between different features. For example, you can calculate the product or ratio of two features to capture potential synergistic effects.

9. Statistical Aggregations:

    Compute statistical aggregations like mean, median, standard deviation, or percentile values for features over specific time windows. These aggregations can capture statistical properties of the data.

10. Domain-Specific Features:

    Consider domain-specific features that might be relevant to stock trading, such as economic indicators (e.g., interest rates, GDP growth) if available.

#### Data Preprocessing

1. One-Hot Encoding or Label Encoding:

    If you have categorical features like imbalance_buy_sell_flag, you can use one-hot encoding or label encoding to represent them numerically for model compatibility.

2. Normalization and Scaling:

    Standardize or normalize your features to have zero mean and unit variance, especially if you are using models like linear regression.

Carefully handle missing data, either through interpolation or imputation.

**Time based splitting**:
when splitting training and testing datasets, make sure to keep time sequence intact.

### Model Selection

- LightGBM
- XGBoost
- [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), long short-term memory, recurrent neural network (RNN)

### Evaluation

Metrics?

Mean Absolute Error
