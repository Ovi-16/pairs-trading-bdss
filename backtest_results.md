# Pairs trading backtest

**Training.** Fit each OLS hedge ratio on 250 trading sessions from 25 August 2025 through 21 August 2026. AVGO/QCOM had a hedge ratio of 0.4110 and an Engle–Granger p-value of 0.034, below the 0.05 cutoff. Both numbers came from the training data.

**Test.** Held each hedge ratio fixed for the 60 trading sessions from 24 August through 16 November 2026. Of 95 pairs with usable data, nine passed the training screen and traded. Three made a net profit. The ranking uses returns from this test window.

| Rank | Pair | Gross return | Net return | Max drawdown | Trades |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | AVGO / QCOM | +17.03% | +16.51% | −4.33% | 4 |
| 2 | AVGO / INTC | +16.26% | +15.75% | −3.67% | 4 |
| 3 | CL / CHD | +0.83% | +0.58% | −1.47% | 2 |
| 4 | MRK / JNJ | ≈0.00% | −0.38% | −3.84% | 3 |
| 5 | KDP / MNST | −2.80% | −3.08% | −7.26% | 2 |
| 6 | NVDA / INTC | −4.87% | −5.21% | −19.62% | 3 |
| 7 | JPM / BAC | −4.88% | −5.27% | −6.59% | 3 |
| 8 | NVDA / AVGO | −12.67% | −12.99% | −21.05% | 3 |
| 9 | NVDA / AMD | −14.38% | −14.67% | −28.56% | 3 |

## Trading rules and costs

- **Entry.** Compare the closing spread with its rolling 20-session mean. Short the spread above +1.8 standard deviations or go long below −1.8.
- **Mean crossing and execution.** Exit a long spread when it reaches or rises above its rolling mean. Exit a short spread when it reaches or falls below the mean. Signals use the close, and trades execute at the next session’s open. Any remaining position is closed at the final test session’s open.
- **Trading cost.** Five basis points equals 0.05% of the combined absolute value of the two legs each time the trade opens or closes. That is $5 to enter and $5 to exit a $10,000 gross position if its size stays the same.
- **Short borrow.** The model charges 1% per year on the dollar value of the short leg, prorated by calendar days held. This represents the fee for borrowing shares to sell short.
- **Daily target weights.** Each day’s return uses the same target proportions for the two legs. In practice, price moves change those proportions. Trading to restore them creates extra turnover, and the model does not charge for those adjustments.
