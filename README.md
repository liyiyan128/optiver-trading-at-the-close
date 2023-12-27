# Optiver - Trading at the Close

Contents:

- [Optiver - Trading at the Close](#optiver---trading-at-the-close)
  - [Background](#background)
    - [Dataset](#dataset)
  - [Strategy](#strategy)
    - [Feature Engineering](#feature-engineering)

## Background

Each trading day on the NASDAQ Stock Exchange ends with the NASDAQ Closing Auction.

During the last ten minutes of a trading session on the Nasdaq exchange, market makers such as Optiver combine traditional order book trading with price bidding data. The ability to integrate information from both sources is crucial in order to provide the optimal price for all market participants.

### Dataset

The [dataset](https://www.kaggle.com/competitions/optiver-trading-at-the-close/data)
contains historic data for the daily ten minute closing auction on the NASDAQ stock exchange.
The **goal** is to predict the future price movements of stocks relative to the price future price movement of a synthetic index composed of NASDAQ-listed stocks.

The features (columns) include

`stock_id`, `data_id`, `imbalance_size`, `imbalance_buy_sell_flag`, `reference_price`, `matched_size`, `far_price`, `near_price`, `[bid/ask]_price`, `[bid/ask]_size`, `wap`, `seconds_in_bucket`

## Strategy

**Feature is all you need**. The primary focus in this competition has been on feature engineering, which is considered as the key to alpha discovery.
The baseline model gradient boosting using LightGBM with feature engineering can achieve a competitive public board score without too much effort in tuning model parameters.
In comparison, baseline LightGBM without feature engineering scores even lower than making constant zero value predictions.

### Feature Engineering

Our feature engineering process absorbs many ideas from the code and the discussion board.

**Basic features**

We first compute some common financial statistics and indicators (e.g. bid-ask spread, trading volume) to reflect liquidity, volatility, pressure, urgency etc.

**Imbalance features**

This idea comes from various notebooks from the [code](https://www.kaggle.com/competitions/optiver-trading-at-the-close/code) and [discussion](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion) board.

From empirical experience, the imbalance features can bring significant improvement in model prediction ability.

- Doublet imbalance ratios:

    Take two feature columns $x$ and $y$ from the prices and sizes feature group, compute $(x - y) / (x + y)$.

- Triplet imbalances:

    Take three feature columns from the prices and sizes feature group, compute (max - mid) / (mid - min) (here min, mid, max are computed row-wise). Thanks again to the Kaggle community, this process is parallelised using numba by the [community work](https://www.kaggle.com/code/lblhandsome/optiver-robust-best-single-model/notebook).

Personally, I find some imbalance ratios difficult to interpret and have little meaning in finance. However, these combinations of features decorrelate the underlying complex information carried in the original data to some extent.

**Lagged features**

Use `shift`, `pct_change` in Pandas to compute lagged features grouped by different feature columns with various window periods.

**Statistical aggregations**

Compute various statistics (mean, standard deviation, skew, kurt, max) for the prices and sizes feature group.

**Temporal features**

Based on provided date_id, seconds_in_bucket, we create features indicating days of the week, seconds/minute in the closing auction.

**Stock specific features**

Group by stock_id, bid_price, ask_price, we create global stock specific features such as median_size, std_price.

**Synthetic index based features**

The competition goal is to predict the future price movements of stocks relative to the price future price movement of a synthetic index composed of NASDAQ-listed stocks.

It turns out that the weights of the synthetic index can be rebuilt by applying linear regression on the stock and index return. See [Weights of the Synthetic Index](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/442851).

Based on synthetic index weights, we create features such as stock_weights (map stock_id by weights), weighted_wap (re-weighted wap).

Group by time_id, weighted_wap, further features such as index wap can be created.

**Memory optimisation**

This is also community work aimed at improving performance and reducing memory-related issues. The original dataset contains various data types (`int8`, `int16`, `float32` etc.). The dataset storage can be optimised by converting columns to the most memory-efficient data types.

See [Memory Optimization Function with Data Type Conversion](https://www.kaggle.com/code/zulqarnainali/lgb-fine-tuned-explained#%F0%9F%9A%80-Memory-Optimization-Function-with-Data-Type-Conversion-%F0%9F%A7%B9).