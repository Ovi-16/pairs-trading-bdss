"""Scan U.S. stock pairs with a frozen OLS hedge and a recent test window."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


GROUPS = (
    ("KO", "PEP", "KDP", "MNST"),
    ("XOM", "CVX", "COP", "EOG", "OXY"),
    ("JPM", "BAC", "WFC", "C", "GS", "MS"),
    ("HD", "LOW", "TGT", "WMT", "COST"),
    ("V", "MA", "AXP", "DFS"),
    ("UNP", "CSX", "NSC"),
    ("UPS", "FDX"),
    ("DAL", "UAL", "LUV", "AAL"),
    ("GM", "F", "TM"),
    ("GOOG", "GOOGL", "META"),
    ("NVDA", "AMD", "AVGO", "QCOM", "INTC"),
    ("DHI", "LEN", "PHM", "TOL"),
    ("PG", "CL", "KMB", "CHD"),
    ("MRK", "PFE", "BMY", "ABBV", "JNJ"),
    ("TJX", "ROST", "BURL"),
)
TRAIN_SESSIONS = 250
LOOKBACK = 20
ENTRY_Z = 1.8


def download(symbol, through):
    end = (through + pd.Timedelta(days=1)).tz_localize("America/New_York")
    start = end - pd.DateOffset(years=2)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={int(start.timestamp())}&period2={int(end.timestamp())}&interval=1d"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=20) as response:
        result = json.load(response)["chart"]["result"][0]
    index = (
        pd.to_datetime(result["timestamp"], unit="s", utc=True)
        .tz_convert("America/New_York").tz_localize(None).normalize()
    )
    indicators = result["indicators"]
    quote = pd.DataFrame(indicators["quote"][0], index=index)
    close = pd.Series(indicators["adjclose"][0]["adjclose"], index=index)
    return pd.DataFrame({"close": close, "open": quote["open"] * close / quote["close"]})


def run_pair(closes, opens, a, b, sessions, cost_bps, borrow_bps):
    log_prices = np.log(closes[[a, b]])
    train = log_prices.iloc[:-sessions]
    fit = sm.OLS(train[a], sm.add_constant(train[b])).fit()
    beta = float(fit.params[b])
    if beta <= 0:
        return None
    p_value = float(coint(train[a], train[b])[1])
    if not np.isfinite(p_value) or p_value >= 0.05:
        return None

    spread = log_prices[a] - fit.params["const"] - beta * log_prices[b]
    rolling = spread.rolling(LOOKBACK)
    z_scores = ((spread - rolling.mean()) / rolling.std()).iloc[-sessions:]
    test_opens = opens[[a, b]].iloc[-sessions:]
    open_returns = (test_opens.shift(-1) / test_opens - 1).to_numpy()
    weights = np.array([1.0, -beta]) / (1 + beta)
    # Use the previous close's signal at this open. Close any position at
    # the final open so both trading legs' exit costs enter realized P&L.
    desired = position = 0
    equity = gross_equity = 1.0
    path = [equity]
    trades = []
    for i, (day, z) in enumerate(z_scores.items()):
        last = i == sessions - 1
        target = 0 if last else desired
        if target != position:
            if target:
                trades.append([day.date().isoformat(), None, target, equity])
            else:
                trades[-1][1] = day.date().isoformat()
        gross = 0.0 if last else target * float(open_returns[i] @ weights)
        cost = abs(target - position) * cost_bps / 10_000
        short_weight = beta / (1 + beta) if target > 0 else 1 / (1 + beta)
        days_held = 0 if last else (test_opens.index[i + 1] - day).days
        borrow = abs(target) * short_weight * borrow_bps / 10_000 * days_held / 365
        equity *= 1 + gross - cost - borrow
        gross_equity *= 1 + gross
        path.append(equity)
        if target != position and target == 0:
            trades[-1][3] = equity / trades[-1][3] - 1
        position = target
        desired = position
        # Exit when the spread crosses its rolling mean. An exact zero
        # check would leave most trades open after they cross the mean.
        if position and position * z >= 0:
            desired = 0
        elif position == 0 and abs(z) > ENTRY_Z:
            desired = -1 if z > 0 else 1

    drawdown = np.min(path / np.maximum.accumulate(path) - 1)
    return a, b, beta, p_value, gross_equity - 1, equity - 1, drawdown, trades


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", required=True, type=pd.Timestamp,
                        help="Last eligible completed US market date, YYYY-MM-DD")
    parser.add_argument("--sessions", type=int, default=60)
    parser.add_argument("--cost-bps", type=float, default=5)
    parser.add_argument("--borrow-bps", type=float, default=100,
                        help="Annual short borrow assumption in basis points")
    args = parser.parse_args()
    if args.sessions < 3 or min(args.cost_bps, args.borrow_bps) < 0:
        parser.error("Use at least three test sessions and nonnegative costs.")

    pairs = [pair for group in GROUPS for pair in combinations(group, 2)]
    symbols = sorted({symbol for pair in pairs for symbol in pair})
    with ThreadPoolExecutor(max_workers=6) as pool:
        downloads = {symbol: pool.submit(download, symbol, args.through) for symbol in symbols}
        prices = {}
        for symbol, future in downloads.items():
            try:
                prices[symbol] = future.result()
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                print(f"Skipping {symbol}: {error}")

    length = TRAIN_SESSIONS + args.sessions
    calendar = pd.DatetimeIndex(sorted({day for frame in prices.values() for day in frame.index}))
    if len(calendar) < length:
        parser.exit(1, "Insufficient price history.\n")
    calendar = calendar[-length:]
    clean = {}
    for symbol, frame in prices.items():
        aligned = frame.reindex(calendar)
        if np.isfinite(aligned).all().all() and (aligned > 0).all().all():
            clean[symbol] = aligned
        else:
            print(f"Skipping {symbol}: missing or invalid prices in the required window.")

    if not clean:
        parser.exit(1, "No complete stock histories.\n")
    closes = pd.DataFrame({symbol: frame["close"] for symbol, frame in clean.items()})
    opens = pd.DataFrame({symbol: frame["open"] for symbol, frame in clean.items()})
    eligible = [(a, b) for a, b in pairs if a in clean and b in clean]
    results = [
        result for a, b in eligible
        if (result := run_pair(closes, opens, a, b, args.sessions,
                              args.cost_bps, args.borrow_bps)) is not None
    ]
    traded = sorted((r for r in results if r[-1]), key=lambda r: r[5], reverse=True)
    winners = [r for r in traded if r[5] > 0]
    print(f"Train: {calendar[0].date()} to {calendar[-args.sessions - 1].date()}")
    print(f"Test:  {calendar[-args.sessions].date()} to {calendar[-1].date()}")
    print(f"Pairs with data: {len(eligible)}/{len(pairs)}; passed training screen: "
          f"{len(results)}; traded: {len(traded)}; profitable: {len(winners)}")
    for a, b, beta, p, gross, net, dd, trades in traded[:5]:
        print(f"{a}/{b}: beta={beta:.4f}, train p={p:.4g}, gross={gross:+.2%}, "
              f"net={net:+.2%}, max drawdown={dd:.2%}, trades={len(trades)}")
    if winners:
        a, b, *_, trades = winners[0]
        print(f"Trades for {a}/{b} (entry open, exit open, side, net return):")
        for entry, exit_date, side, net in trades:
            print(f"  {entry}  {exit_date}  {'long spread' if side > 0 else 'short spread'}  {net:+.2%}")
    else:
        print("No profitable pair under these assumptions.")


if __name__ == "__main__":
    main()
