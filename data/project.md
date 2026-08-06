# The Wall Street Quants Course Project

## Statistical Arbitrage in Cryptocurrencies

This project is designed to give you a chance to apply the concepts you've learned in class so far. It's also meant to provide an example of a research project that might be undertaken at a Quant Hedge Fund.

---

## Project Goal

Statistical arbitrage is a class of strategies that tries to discover price-volume patterns that predict returns. It is one of the most popular and successful quantitative hedge-fund strategies.

Cryptocurrency markets are still relatively new and should be fertile grounds for finding market inefficiencies using statistical arbitrage techniques. The two main patterns exploited in statistical arbitrage are **momentum** and **reversal**.

**The goal of this project is to research profitable momentum and/or reversal strategies in crypto.**

---

## Research Outline

Please watch the momentum & reversal course videos if you haven't already — they will serve as a rough research outline for this project.

In the videos, we provide guidance on how to find reversal and momentum, as well as specific thoughts on applications to crypto. We've recapped these points below.

Please use these as starting points for your research. You can either go deep on one of the points or explore many. Also feel free to pursue any ideas you may be excited about that are not listed here.

### How to Find Momentum

1. **Time Horizon**
   - Longer time horizons generally lead to momentum.
   - Test different time horizons and see where momentum might exist.

2. **New Information / Activity**
   - Times of heightened activity coupled with new information favor momentum.
   - Apply indicators of activity / new info (e.g. Twitter activity, trading volume) to find stronger momentum.

3. **Seasonality**
   - Similar seasons tend to show momentum.
   - What are the relevant seasons in crypto and do they show momentum? As we discussed, it could be worthwhile to explore times when institutions vs. retail trade (e.g. weekdays vs. weekends, day vs. night).

4. **Investment Themes**
   - Investment "themes" or "styles" tend to show momentum.
   - We discussed some relevant themes in crypto. Do these or other themes you can think of show momentum?

5. **Technical Plays**
   - Sometimes, predictable mechanical rebalancing leads to momentum.
   - Given the institutional crypto players and their trading schedules (e.g. usually during work-hours), is it possible to front-run them?

### How to Find Reversal

1. **Time Horizon**
   - Shorter time horizons generally lead to reversal.
   - Test different time horizons and see where reversal might exist.

2. **Uninformed Trading**
   - Uninformed (liquidity-driven) trades reverse more.
   - Apply indicators of activity / new info (e.g. Twitter activity, trading volume) and isolate cases of lower activity / info to strengthen reversal.
   - Potentially draw upon ideas mentioned in the "Fire Sale" crypto video, which relied on uninformed trading during liquidations to find reversal.

3. **Correlation**
   - Security A − (Something Correlated to It) is more mean-reverting.
   - Find either correlated pairs or baskets of crypto assets as discussed in the video on reversal/correlation.

4. **Macro**
   - There is more reversal in times when there's more volatility and dislocation in the macro environment.
   - Test if this is true using the indicators of volatility / dislocation we discussed in class (implied volatility, realized volatility, return dispersion, pairwise correlation).

---

## Data

Cryptocurrency price-volume data is freely available. Please refer to our **"PriceData"** lecture from week 3 on how to grab crypto price-volume data.

---

## Backtesting

To start, please use the **"unconstrained"** style of backtests we introduced in the backtesting section of the course. If needed, we can move onto other types of backtests later.

---

## Execution / Slippage

Cryptocurrencies can have commissions of ~7 bps. While total slippage is unknown and will depend on the trader's volume as well, let's assume another 13 bps. So total all-in execution costs will be **20 bps** for market orders. Limit orders will just have the **7 bps** of commissions.

It's possible you will also have to try to alter trading to improve execution. Please refer to our **Execution – Trading** video for guidelines.

---

## Weighting

During your research, it's very possible you find more than one strategy you find compelling. For example, you may find 1 momentum strategy and 1 reversal strategy. Please refer to our videos on weighting for guidelines on how to combine them appropriately.

---

## Performance Evaluation

Please provide the key performance evaluation metrics we've discussed in class so far. This includes:

- Returns
- Volatility
- Sharpes
- Max drawdowns
- Alpha / beta

---

## Additional Evaluation Objectives

| Objective | Why it matters |
|---|---|
| Generalization | Predictive usefulness |
| Calibration | Probabilistic reliability |
| Robustness | Stability under perturbations |
| Distribution shift tolerance | Real-world drift |
| Interpretability | Human trust / debugging |
| Latency | Deployment constraints |
| Memory / compute efficiency | Cost |
| Fairness | Regulatory / social constraints |
| Privacy | Legal / security concerns |
| Controllability | Alignment with system goals |
| Sample efficiency | Limited data regimes |
