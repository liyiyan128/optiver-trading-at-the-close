# Optivers - Trading at the Close

Final ten minutes of the trading day, often characterised by heightened volatility and rapid price fluctuations

In the last ten minitues of Nasdaq exchange trading session, market makers like Optiver merge traditional order book data with auction book data.

**Goal**:
predict the stock closing price movements for hundreds of Nasdaq listed stocks
using data from the order book and the closing auction of the stock

Information from teh auction can be used to adjust prices, assess supply and demand dynamics, and identify trading opportunities

## Dataset

Historic data for the daily ten nimute closing auction on NASDAQ stock exchange.

continuous phase: best bid, best ask, best volume

auction phase: reference price, matched volume, imbalnce

On-close orders (MOC, LOC, IO) together with orders from the continuous phase contribute to auction volume and imbalance.
The final price is determined to maximize the matched volume. Therefore, the volume to buy and to sell moves the final price.

## Background Knowledge

Bid_ask_spread = best_offer / best_bid - 1

WAP = (bid_price_1 * ask_size_1 + ask_price_1 * bid_size_1) / (bid_size_1 + ask_size_1)

log return $r_{t1, t2} = log({\frac{S_t2}{S_t1}})$

volatility $\sigma = \sqrt{{\Sigma}_{t}r^{2}_{t-1, t}}$

## Plan

