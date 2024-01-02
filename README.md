# Imperial x Optiver | Public Board No. 55 - Silver Medal

## Optiver - Trading at the Close

Contents:

- [Imperial x Optiver | Public Board No. 55 - Silver Medal](#imperial-x-optiver--public-board-no-55---silver-medal)
  - [Optiver - Trading at the Close](#optiver---trading-at-the-close)
  - [Background](#background)
  - [Dataset](#dataset)
  - [Feature Engineering](#feature-engineering)
  - [LightGBM \& MLP](#lightgbm--mlp)
    - [LightGBM](#lightgbm)
    - [MLP (multilayer perception)](#mlp-multilayer-perception)

## Background

Each trading day on the NASDAQ Stock Exchange ends with the NASDAQ Closing Auction.

During the last ten minutes of a trading session on the Nasdaq exchange, market makers such as Optiver combine traditional order book trading with price bidding data. The ability to integrate information from both sources is crucial in order to provide the optimal price for all market participants.

## Dataset

The [dataset](https://www.kaggle.com/competitions/optiver-trading-at-the-close/data)
contains historic data for the daily ten minute closing auction on the NASDAQ stock exchange.
The **goal** is to predict the future price movements of stocks relative to the price future price movement of a synthetic index composed of NASDAQ-listed stocks.

The features (columns) include

`stock_id`, `data_id`, `imbalance_size`, `imbalance_buy_sell_flag`, `reference_price`, `matched_size`, `far_price`, `near_price`, `[bid/ask]_price`, `[bid/ask]_size`, `wap`, `seconds_in_bucket`

## Feature Engineering

**Feature is all you need**. The primary focus in this competition has been on feature engineering, which is considered as the key to alpha discovery.

The baseline model gradient boosting using LightGBM with feature engineering can achieve a competitive public board score without too much effort in tuning model parameters.
In comparison, baseline LightGBM without feature engineering scores even lower than making constant zero value predictions.

Our feature engineering process absorbs many ideas from the [code](https://www.kaggle.com/competitions/optiver-trading-at-the-close/code) and [discussion](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion) board.

**Basic features**

We first compute some common financial statistics and indicators (e.g. bid-ask spread, trading volume) to reflect liquidity, volatility, pressure, urgency etc.

**Imbalance features**

This idea comes from various notebooks from the code and the discussion board.

From empirical experience, the imbalance features can bring significant improvement in model prediction ability.

- Doublet imbalance ratios:

    Take two feature columns $x$ and $y$ from the prices and sizes feature group, compute $(x - y) / (x + y)$.

- Triplet imbalances:

    Take three feature columns from the prices and sizes feature group, compute (max - mid) / (mid - min) (here min, mid, max are computed row-wise). Thanks again to the Kaggle community, this process is parallelised using numba by the [community work](https://www.kaggle.com/code/lblhandsome/optiver-robust-best-single-model/notebook).

Personally, I find some imbalance ratios difficult to interpret and have little meaning in finance. Some features are just trying out possible combinations in the hope of improving prediction ability. Based on public scores, the new features created indeed decorrelate the underlying complex information carried in the original data to some extent.

**Lagged features**

Use `diff`, `shift`, `pct_change` in Pandas to compute lagged features for various features (prices, sizes) grouped by stock_id with various window periods (1, 2, 3, 10).

**Statistical aggregations**

Compute various statistics (mean, standard deviation, skew, kurt, max) for the prices and sizes feature group.

**Temporal features**

Based on provided date_id, seconds_in_bucket, we create features indicating days of the week, seconds/minute during the closing auction.

**Stock specific features**

Group by stock_id, bid_price, ask_price, we create global stock-specific features such as median_size, std_price. These global stock-specific features reflect global market trends.

**Synthetic index based features**

The competition goal is to predict the future price movements of stocks relative to the price future price movement of a synthetic index composed of NASDAQ-listed stocks.

It turns out that the weights of the synthetic index can be rebuilt by applying linear regression on the stock and index return. See [Weights of the Synthetic Index](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/442851).

- Based on synthetic index weights, we create features such as stock_weights (map stock_id by weights), weighted_wap (re-weighted wap).

- Group by time_id, weighted_wap, further features such as index wap can be created.

**Memory optimisation**

This is also community work aimed at improving performance and reducing memory-related issues. The original dataset contains various data types (`int8`, `int16`, `float32` etc.). The dataset storage can be optimised by converting columns to the most memory-efficient data types.

See [Memory Optimization Function with Data Type Conversion](https://www.kaggle.com/code/zulqarnainali/lgb-fine-tuned-explained#%F0%9F%9A%80-Memory-Optimization-Function-with-Data-Type-Conversion-%F0%9F%A7%B9).

## LightGBM & MLP

We produced two solutions. Single models are usually less prone to overfitting compared to ensemble models. However, ensemble models may provide better generalisation ability.

1. [LightGBM Robust Single Model](https://github.com/liyiyan128/optiver-trading-at-the-close/blob/d5bc5e3458820dfee7bab1afe52d99306df3c98f/code/lgbm.ipynb).
2. [LightGBM & MLP Ensemble Model](https://github.com/liyiyan128/optiver-trading-at-the-close/blob/d5bc5e3458820dfee7bab1afe52d99306df3c98f/code/lgbm_mlp.ipynb).

In this competition, models are trained on historic data and are tested on latest real-time data. Therefore, the model generalisation ability can be crucial.

### LightGBM

LightGBM (light gradient-boosting machine) is a popular gradient boosting framework developed by Microsoft. LightGBM uses a histogram-based algorithm to split data, reducing memory usage and speeding up training. The decision trees are grown in a leaf-wise manner, rather than level-wise in traditional boosting algorithms. This approach can lead to more depth in trees, capturing complex patterns.

**Model training and parameter tuning**

The dataset contains time series data. To avoid data leakage, we adopted time-based split (split data based on specific time points e.g. split_day=435) to separate the dataset into train and valid sets.

To reduce notebook running time, we used pre-trained model weights in the submission notebook.

### MLP (multilayer perception)

A multilayer perceptron (MLP) is a neural network consisting of fully connected neurons. MLPs are able to distinguish data that is not linearly separable.
MLPs serve as a foundation for many sophisticated neural network architectures.
Theoretically, MLPs can approximate any continuous function given enough neurons and layers.

![MLP](https://github.com/liyiyan128/optiver-trading-at-the-close/blob/main/figure_mlp.png)
