# Core Quant Terms for Students

Before walking through the notebook, keep these seven terms handy—you can define them on the fly whenever they appear:

* **Random Walk:** A time series where each step is a random shock added to the previous value ($x_t = x_{t-1} + \epsilon_t$). Like an unseeded drunk walk: its variance grows with time, and it never "remembers" where it started.

* **Stationarity:** A statistical property where a series’ mean and variance stay roughly constant over time. A stationary series fluctuates around a stable baseline; a random walk is non-stationary.

* **Cointegration:** Two non-stationary series ($y_t$ and $x_t$) that each wander like random walks on their own, yet a specific linear combination of them ($y_t - \beta x_t$) is stationary. Think of two people walking randomly in a park while holding a bungee cord between them: each person's path is a random walk, but the distance between them stays bounded.

* **Hedge Ratio (**$\hat{\beta}$**):** The scaling weight used to combine two assets so their common movement cancels out. Estimated via Ordinary Least Squares (OLS) linear regression.

* **Spread (**$s_t$**):** The residual error $y_t - \hat{\beta} x_t$. This is the actual "bungee cord distance" we trade.

* **Half-Life (**$t_{1/2}$**):** How many trading days it takes for a shock in the spread to decay by half. In CS terms, it is the time constant of a first-order decay filter.

* **Basis Points (bps):** $1\text{ bps} = 0.01\% = 0.0001$. Saying transaction costs are 5 bps means paying $0.05\%$ per unit of position changed.

# Section-by-Section Guide to pairs_trading_workshop.ipynb

## Setup Cell: Building a Deterministic Test Universe

In `load_workshop_universe()`, we generate 10 synthetic assets (ASSET_A1 through ASSET_D4) over 750 trading days instead of calling a live market API.

* **Why do this in a workshop?** Live APIs fail on Wi-Fi, rate-limit rooms full of laptops, and hide ground truth. Here, we bake known regimes into the data:

  * ASSET_A1 & ASSET_A2: truly cointegrated pair (fast reversion).

  * ASSET_B1 & ASSET_B2: truly cointegrated pair (moderate reversion).

  * ASSET_C1 & ASSET_C2: high correlation, not cointegrated.

* **Key line to point out:** `log_prices = np.log(prices)`. Taking logs turns multiplicative percentage returns into additive differences, so linear regression ($y - \beta x$) models percentage relationships rather than dollar-price scale.

## Section 1: Why Correlation Fails (`compare_correlation_vs_spread`)

Students almost always ask: "Why not just pick the two stocks with the highest Pearson correlation?"

* **What the code does:** Computes Pearson correlation $r$ on `log_prices`, fits an OLS line $y_t = \alpha + \beta x_t + \epsilon_t$, defines `spread = log_df[a] - beta * log_df[b]`, and runs `adfuller(spread)` (the Augmented Dickey-Fuller test, a hypothesis test for whether a series has a unit root / random walk behavior).

* **How to present it:** Compare Pair A vs. Pair C side-by-side. Even when two series share a common sector factor and look visually aligned on the left plot, Pair C's spread on the right drifts away with a large ADF p-value (well above 0.05). If you bet on Pair C reverting to its mean, you are waiting for a random walk to come home.

## 3. How `adfuller(spread)` Works Here

The Augmented Dickey-Fuller (ADF) test is a statistical hypothesis test for whether a time series has a unit root—meaning it behaves like a random walk (non-stationary, where shocks never decay and variance grows over time).

### The Regression Inside ADF and Why the Lagged Terms Are There

Internally, `adfuller(spread)` estimates the following regression using Ordinary Least Squares (OLS):

$$
\Delta s_t = c + \gamma s_{t-1} + \sum_{i=1}^{p} \delta_i \Delta s_{t-i} + \eta_t
$$

where:

* $\Delta s_t = s_t - s_{t-1}$ is today's change in the spread,

* $s_{t-1}$ is yesterday's spread level,

* $\sum_{i=1}^{p} \delta_i \Delta s_{t-i}$ is a sum of past changes (lags of $\Delta s$),

* $\eta_t$ is zero-mean noise.

**Why include** $\sum_{i=1}^{p} \delta_i \Delta s_{t-i}$**?** In real financial data, today's change $\Delta s_t$ often correlates slightly with yesterday's change $\Delta s_{t-1}$ (short-term momentum or microstructure noise). If you leave those lagged $\Delta s$ terms out of the regression, that short-term serial correlation gets dumped into the error term $\eta_t$, which ruins the standard error of $\hat{\gamma}$ and makes the ADF test unreliable. Including the lagged changes cleans up the short-term wiggle so that $\hat{\gamma}$ isolates one specific question: *does the level* $s_{t-1}$ *pull* $\Delta s_t$ *back toward an equilibrium mean?*

### Case 1: What Happens When $\gamma = 0$ (Unit Root / Pure Accumulator)

If $\gamma = 0$, the equation does **not** drop the summation—it simply drops the level term $s_{t-1}$:

$$
\Delta s_t = c + \sum_{i=1}^{p} \delta_i \Delta s_{t-i} + \eta_t
$$

Look at what variables remain on the right-hand side: only the constant $c$, past changes ($\Delta s_{t-1}, \Delta s_{t-2}, \dots$), and noise ($\eta_t$). The spread level $s_{t-1}$ is completely absent. In CS terms, $s_t$ acts as a **pure accumulator**:

* **Suppose** $p = 0$ (no lagged $\Delta s$ terms): Then $\Delta s_t = c + \eta_t$, which rearranges to:

  $$
  s_t = s_{t-1} + c + \eta_t
  $$

  Every random shock $\eta_t$ permanently shifts $s_t$. Nothing pulls it back.

* **Suppose** $p > 0$ (e.g., $\Delta s_t = c + \delta_1 \Delta s_{t-1} + \eta_t$): Even though past changes influence today's change, there is still no restoring force tied to the level $s_{t-1}$. If a large shock $\eta_{100}$ hits on day 100, $s_t$ jumps to a new baseline and stays there—its variance keeps growing over time (non-stationary).

### Case 2: Why $\gamma < 0$ Pulls the Spread Back to Its Mean

Now examine what happens when $\gamma < 0$. First, rewrite the ADF equation in level form by moving $s_{t-1}$ to the right-hand side:

$$
s_t - s_{t-1} = c + \gamma s_{t-1} + \sum_{i=1}^{p} \delta_i \Delta s_{t-i} + \eta_t
$$

$$
s_t = c + (1 + \gamma) s_{t-1} + \sum_{i=1}^{p} \delta_i \Delta s_{t-i} + \eta_t
$$

#### Finding the Long-Run Equilibrium $\mu$ (Using Expectations, Not Erasure)

In a stochastic process, we do not pretend noise $\eta_t$ or lagged changes vanish by hand-waving. Instead, we take the expectation $\mathbb{E}[\cdot]$—the average over all possible random realizations. Suppose $s_t$ is stationary with a constant long-run mean $\mu = \mathbb{E}[s_t]$, which implies $\mathbb{E}[s_t] = \mathbb{E}[s_{t-1}] = \mu$.

Applying $\mathbb{E}[\cdot]$ to both sides of the original ADF equation gives:

$$
\mathbb{E}[\Delta s_t] = \mathbb{E}[c] + \gamma \mathbb{E}[s_{t-1}] + \sum_{i=1}^{p} \delta_i \mathbb{E}[\Delta s_{t-i}] + \mathbb{E}[\eta_t]
$$

Under stationarity, each term evaluates cleanly:

* $\mathbb{E}[\Delta s_t] = \mathbb{E}[s_t - s_{t-1}] = \mu - \mu = 0$ (on average, the level is neither growing nor shrinking),

* $\mathbb{E}[\Delta s_{t-i}] = \mathbb{E}[s_{t-i} - s_{t-i-1}] = \mu - \mu = 0$ for every lag $i$,

* $\mathbb{E}[\eta_t] = 0$ (by definition of zero-mean error noise),

* $\mathbb{E}[c] = c$.

Thus, the lagged terms and noise have zero expectation under stationarity, leaving:

$$
0 = c + \gamma \mu + 0 + 0 \implies \mu = -\frac{c}{\gamma}
$$

#### Keeping All Terms Intact: How the System Actually Moves

Substituting $c = -\gamma \mu$ back into the original ADF equation—while keeping $\sum_{i=1}^p \delta_i \Delta s_{t-i}$ and $\eta_t$ intact—gives:

$$
\Delta s_t = -\gamma \mu + \gamma s_{t-1} + \sum_{i=1}^{p} \delta_i \Delta s_{t-i} + \eta_t
$$

$$
\Delta s_t = \underbrace{\gamma (s_{t-1} - \mu)}_{\text{1. Level restoring force}} + \underbrace{\sum_{i=1}^{p} \delta_i \Delta s_{t-i}}_{\text{2. Short-run momentum / lag term}} + \underbrace{\eta_t}_{\text{3. Random shock}}
$$

Look at the sign of the restoring force $\gamma(s_{t-1} - \mu)$ when $\gamma < 0$ (for example, $\gamma = -0.1$):

* **When** $s_{t-1} > \mu$ **(spread is above its mean):** $(s_{t-1} - \mu) > 0$, so multiplying by $\gamma = -0.1$ yields a negative pull:

  $$
  \gamma (s_{t-1} - \mu) = (-0.1) \times (\text{positive}) < 0
  $$

  Today's change $\Delta s_t$ is pushed downward toward $\mu$.

* **When** $s_{t-1} < \mu$ **(spread is below its mean):** $(s_{t-1} - \mu) < 0$, so multiplying by $\gamma = -0.1$ yields a positive pull:

  $$
  \gamma (s_{t-1} - \mu) = (-0.1) \times (\text{negative}) > 0
  $$

  Today's change $\Delta s_t$ is pushed upward toward $\mu$. *(Note:* $\Delta s_t$ *is the step size, not* $s_t$ *itself. If* $\mu = 5.0$ *and* $s_{t-1} = 4.2$*, a positive* $\Delta s_t$ *of* $+0.15$ *moves* $s_t$ *from* $4.2$ *to* $4.35$*, closer to* $5.0$*.)*

Because all three terms act together, $\gamma < 0$ does **not** mean $s_t$ moves monotonically in a straight line toward $\mu$ every single day:

1. **The level term** $\gamma(s_{t-1} - \mu)$ acts like a spring pulling $s_t$ back toward $\mu$.

2. **The lag sum** $\sum_{i=1}^p \delta_i \Delta s_{t-i}$ carries short-term inertia. If the spread was climbing fast ($\Delta s_{t-1} > 0$) and $\delta_1 > 0$, inertia can push $s_t$ past $\mu$ for a day or two before the restoring force pulls it back—producing damped oscillation around $\mu$.

3. **The noise term** $\eta_t$ injects random shocks at every step. On any given day, a large shock $\eta_t$ can overpower $\gamma(s_{t-1} - \mu)$ and knock $s_t$ temporarily further away from $\mu$.

Overall, the spring keeps the noisy, bumpy path bounded around $\mu$ instead of drifting off to $\pm\infty$. That is why in the notebook's Pair A plot, the spread looks like a bouncy wave crossing the dashed mean line repeatedly, whereas Pair C wanders off without any spring pulling it back.

### Case 3: Why Can $\gamma > 0$ Happen? (Diverging Series)

Looking back at the level-form multiplier $(1 + \gamma)$ on $s_{t-1}$:

* **If** $\gamma = 0$**:** $1 + \gamma = 1$ (unit root / random walk).

* **If** $-2 < \gamma < 0$**:** $\vert 1 + \gamma \vert < 1$ (deviations decay over time; mean-reverting / stationary).

* **If** $\gamma > 0$**:** $1 + \gamma > 1$ (deviations are amplified at every step).

There are two reasons you might encounter $\hat{\gamma} > 0$:

1. **In a true explosive/diverging process:** Like an unstable feedback loop in control theory with loop gain $> 1$, any slight deviation from baseline pushes $s_t$ even further away in the same direction, causing it to drift or blow up exponentially—the exact opposite of mean reversion.

2. **In finite-sample estimation:** Even if the true $\gamma$ is $0$ (a pure random walk) or slightly negative, running OLS on a finite window of $T = 750$ trading days produces a noisy estimate $\hat{\gamma}$. Because a random walk wanders in large swings, a 750-day slice may happen to trend upward, causing OLS to fit $\hat{\gamma} > 0$. In pairs trading, $\hat{\gamma} > 0$ is an immediate red flag that the spread is diverging rather than reverting.

### Reading `adf_stat` and `p_val`

In `adfuller(spread)`, we run a **one-sided test on the left tail** ($\gamma < 0$):

* **Null hypothesis (**$H_0$**):** $\gamma = 0$ — the spread has a unit root (it is a non-stationary random walk; shocks never decay).

* **Alternative hypothesis (**$H_1$**):** $\gamma < 0$ — the spread is stationary / mean-reverting (shocks decay back toward a mean).

Because $H_1$ tests strictly for $\gamma < 0$, any estimate $\hat{\gamma} \ge 0$ provides zero evidence for mean reversion.

* **`adf_stat`:** The test statistic (analogous to a t-statistic) for $\gamma < 0$. The more negative `adf_stat` is, the stronger the statistical evidence of a restoring pull toward the mean.

* **`p_val`:** Answers one specific question: *"Suppose the spread were actually a non-stationary random walk (*$\gamma = 0$*). If we ran this experiment again, what is the probability that random noise alone would produce an ADF test statistic at least as negative as the one we just saw?"*

  * **Small `p_val` (e.g., `< 0.05`):** Seeing a negative $\hat{\gamma}$ that strong is unlikely to happen by random-walk fluke. We reject $H_0$ and treat the spread as stationary / mean-reverting.

  * **Large `p_val` (e.g., `0.73`):** Random walk noise easily produces test statistics like this. We fail to reject $H_0$—we lack statistical evidence that the spread snaps back. *(Note: failing to reject* $H_0$ *does not prove* $H_0$ *is true; it simply means there is no statistically significant mean reversion in that sample.)*

## Quant & Backtesting Terms in Section 3

Before stepping through the code, here are the new terms you will encounter in this section:

* **Lookahead Bias:** Using information from the future (such as tomorrow’s price or full-sample statistics) to decide today’s trade. In CS terms, it is data leakage between train and test steps.

* **Rolling Z-Score (**$z_t$**):** How many standard deviations today’s spread is away from its recent moving average:

  $$
  z_t = \frac{s_t - \mu_{\text{roll}, t}}{\sigma_{\text{roll}, t}}
  $$

  Think of it as normalizing a sensor reading against a sliding window of recent history.

* **Long Spread vs. Short Spread:**

  * **Long spread (+1):** Buy `leg_a` and short $\hat{\beta}$ units of `leg_b`. You profit if `leg_a` rises relative to `leg_b` (the spread goes up).

  * **Short spread (-1):** Short `leg_a` and buy $\hat{\beta}$ units of `leg_b`. You profit if `leg_a` falls relative to `leg_b` (the spread goes down).

* **Execution Lag:** A delay between computing a signal and actually holding the position. If you compute a signal at market close on day $t$, you can only hold it for day $t+1$’s return.

* **Turnover:** How much your position changes from one step to the next, measured as $\vert pos_t - pos_{t-1} \vert$. Going from `0` to `+1` is turnover `1`; flipping from `+1` to `-1` is turnover `2`.

* **Sharpe Ratio:** Return divided by volatility (standard deviation of returns), usually annualized by multiplying by $\sqrt{252}$ (since there are roughly 252 trading days in a year). Higher means more return per unit of risk.

* **Maximum Drawdown:** The largest peak-to-trough percentage drop in your cumulative equity curve—essentially the worst loss an investor would have sat through from a high-water mark.

## Block-by-Block Walkthrough of `backtest_pair`

### 1. Function Inputs & Price Representations

```python
def backtest_pair(
    prices_df: pd.DataFrame,
    leg_a: str,
    leg_b: str,
    lookback: int = 30,
    z_entry: float = 2.0,
    z_exit: float = 0.0,
    cost_bps: float = 5.0,) -> pd.DataFrame:
    log_df = np.log(prices_df[[leg_a, leg_b]])
    ret_a = prices_df[leg_a].pct_change().fillna(0.0)
    ret_b = prices_df[leg_b].pct_change().fillna(0.0)
```

* `log_df` is used to build the spread $s_t = \log P_{A,t} - \hat{\beta} \log P_{B,t}$ so the relationship stays scale-invariant.

* `ret_a` and `ret_b` are simple daily percentage returns $\frac{P_t - P_{t-1}}{P_{t-1}}$, with `.fillna(0.0)` on row 0 where no prior price exists. We use percentage returns here because actual PnL (profit and loss) on stocks is driven by percentage price changes.

### 2. Hedge Ratio & Rolling Z-Score

```python
    # Static hedge ratio estimated on initial window to avoid full-sample lookahead
    init_slice = log_df.iloc[:250]
    X_init = sm.add_constant(init_slice[leg_b])
    beta = float(sm.OLS(init_slice[leg_a], X_init).fit().params[leg_b])

    spread = log_df[leg_a] - beta * log_df[leg_b]
    roll_mean = spread.rolling(lookback).mean()
    roll_std = spread.rolling(lookback).std()
    z_score = (spread - roll_mean) / roll_std
```

* **Why `init_slice = log_df.iloc[:250]`?** If you fit `beta` on all 750 days, `beta` encodes information from days 251–749. Fitting on the first 250 days mimics estimating the hedge ratio on historical data first. *(Note: in a strict out-of-sample setup, you would also evaluate on days after 250 or re-fit on a rolling window—a great point to mention in Section 5.)*

* `spread.rolling(lookback).mean()` and `.std()` compute backward-looking windows of length `lookback` (default 30 days). The first `lookback - 1` rows of `z_score` are `NaN` because there isn’t a full window yet.

### 3. Position State Machine

```python
    # State machine for long/short spread positions
    pos = np.zeros(len(spread))
    curr = 0.0
    for i, z in enumerate(z_score.values):
        if np.isnan(z):
            curr = 0.0
        elif curr == 0.0:
            if z > z_entry:
                curr = -1.0  # Short spread: short A, long beta * B
            elif z < -z_entry:
                curr = 1.0   # Long spread: long A, short beta * B
        else:
            if abs(z) <= z_exit:
                curr = 0.0
        pos[i] = curr
```

You can present this directly as a 3-state finite state machine with states `{-1.0, 0.0, +1.0}`:

* **From `0.0` (flat):**

  * If `z > +z_entry` (e.g., `> 1.8`), the spread is unusually high, so we bet it comes down $\rightarrow$ enter `-1.0` (short spread).

  * If `z < -z_entry` (e.g., `< -1.8`), the spread is unusually low, so we bet it goes up $\rightarrow$ enter `+1.0` (long spread).

* **When in a position (`curr != 0.0`):**

  * Hold the position until `abs(z) <= z_exit` (e.g., `z` crosses back near `0.0`), at which point we flatten to `0.0`.

  * Notice why hysteresis (`z_entry = 1.8`, `z_exit = 0.0`) matters: if you entered and exited at the exact same threshold, tiny noise around the boundary would cause rapid chattering.

### 4. Execution Lag, Turnover & Transaction Costs

```python
    positions = pd.Series(pos, index=spread.index).shift(1).fillna(0.0)
    turnover = positions.diff().abs().fillna(positions.abs())

    # Dollar-neutralized leg weights
    gross_ret = positions * (ret_a - beta * ret_b) / (1.0 + abs(beta))
    tx_cost = turnover * (cost_bps * 1e-4)
    net_ret = gross_ret - tx_cost
```

* **`.shift(1)` (Execution lag):** At index `i`, `pos[i]` was decided using `z_score[i]` (which uses close prices up to day `i`). Shifting by 1 means `positions[i]` equals `pos[i - 1]`, so day `i`’s return only uses signals known at day `i - 1`’s close.

* **`turnover`:** `positions.diff().abs()` computes $\vert positions_t - positions_{t-1} \vert$. Every time you enter, exit, or flip sides, you pay `cost_bps * 1e-4` (for 5 bps, that is `0.0005` or `0.05%` per unit of turnover).

* **`gross_ret` formula:** `positions * (ret_a - beta * ret_b) / (1.0 + abs(beta))` computes the return of holding 1 unit of `leg_a` and `-beta` units of `leg_b`, divided by `1.0 + abs(beta)` so the gross notional exposure scales cleanly rather than blowing up when `beta` is large.

### 5. Cumulative Equity Curves

```python
    return pd.DataFrame({
        'spread': spread,
        'z_score': z_score,
        'position': positions,
        'gross_ret': gross_ret,
        'net_ret': net_ret,
        'cum_gross': (1 + gross_ret).cumprod(),
        'cum_net': (1 + net_ret).cumprod(),
    })
```

* `.cumprod()` compounds daily returns $(1 + r_1) \times (1 + r_2) \times \dots \times (1 + r_t)$, turning a series of daily returns into a \$1-initialized equity curve.

## Walkthrough of `summarize_performance`

```python
def summarize_performance(bt: pd.DataFrame, ann_factor: int = 252) -> pd.Series:
    r = bt['net_ret']
    sharpe = np.sqrt(ann_factor) * r.mean() / (r.std() + 1e-12)
    eq = bt['cum_net']
    drawdown = (eq / eq.cummax() - 1.0).min()
    total_return = eq.iloc[-1] - 1.0
    return pd.Series({
        'total_net_return': total_return,
        'ann_sharpe_net': sharpe,
        'max_drawdown_net': drawdown,
    })
```

* **`ann_sharpe_net`:** `r.mean() / r.std()` is the daily Sharpe ratio (assuming risk-free rate $\approx 0$), multiplied by $\sqrt{252}$ to annualize. Adding `1e-12` prevents division by zero if a series is all zeros.

* **`max_drawdown_net`:** `eq.cummax()` tracks the running highest equity value so far. `eq / eq.cummax() - 1.0` gives the percentage drop from that peak at every day, and `.min()` grabs the worst drop (most negative number).

* **`total_net_return`:** `eq.iloc[-1] - 1.0` converts the final equity multiplier (e.g., `1.03`) into net percentage return (`+0.03`, or `+3%`).

## Deep-Dive: Spread Position Mechanics, Hedging, and Price Scale

Let’s unpack three core mechanics directly against the spread definition in `pairs_trading_workshop.ipynb`:

$$
s_t = \log P_{A,t} - \hat{\beta} \log P_{B,t}
$$

In the backtest cell, the raw return of that spread is:

$$
\text{ret}_t = r_{A,t} - \hat{\beta} r_{B,t}
$$

where $r_{A,t}$ and $r_{B,t}$ are daily percentage returns (`.pct_change()`), and `gross_ret` simply scales this by `positions / (1.0 + abs(beta))`.

### 1. Does $+1$ Mean $1$ Unit of $A$ and $-\hat{\beta}$ Units of $B$?

In return/log-price space, yes: holding long $A$ ($+1$) and short $\hat{\beta}$ of $B$ (`-beta`) gives a return proportional to $r_{A,t} - \hat{\beta} r_{B,t}$.

* **Why $\hat{\beta}$ specifically?** From the OLS regression $\log P_{A,t} = \alpha + \beta \log P_{B,t} + \epsilon_t$, $\hat{\beta}$ estimates how much `leg_a` typically moves (in percentage/log terms) for a 1-unit move in `leg_b`. Multiplying `leg_b` by $\hat{\beta}$ puts both legs on the same sensitivity scale so their common movement cancels out.

### 2. Why Short $B$ at All? Couldn't You Just Go Long $A$?

Think of what happens when the whole sector or market moves. Suppose tomorrow a macro shock hits and both $A$ and $B$ drop by $10\%$ ($r_A = -10\%$, $r_B = -10\%$), with $\hat{\beta} = 1$:

* **If you ONLY went long $A$:** You lose $-10\%$, even though $A$ did not fall relative to $B$.

* **If you go long $A$ and short $\hat{\beta}$ of $B$:**

  * Long $A$ gives $-10\%$

  * Short $B$ ($-1 \times (-10\%)$) gives $+10\%$

  * **Combined return:** $-10\% + 10\% = 0\%$

Now look at the cases where you **do** make money on a long spread ($+1$):

* **$B$ stays flat ($r_B = 0\%$), $A$ rises ($r_A = +3\%$):** Long $A$ gives $+3\%$, short $B$ gives $0\%$ $\rightarrow$ you make $+3\%$.

* **Both fall, but $A$ falls less ($r_A = -1\%$, $r_B = -5\%$):** Here $A$ rose relative to $B$ (the spread $s_t$ went up). Long $A$ gives $-1\%$, short $B$ gives $-1 \times (-5\%) = +5\%$ $\rightarrow$ net $+4\%$.

* **Both rise, but $A$ rises more ($r_A = +6\%$, $r_B = +2\%$):** Long $A$ gives $+6\%$, short $B$ gives $-2\%$ $\rightarrow$ net $+4\%$.

That is the core trade-off: going long $A$ alone is a bet on the market/sector going up; going long $A$ + short $\hat{\beta} B$ strips out the shared market/sector move and isolates relative performance ($r_A - \hat{\beta} r_B$).

### 3. Does This Assume $A$’s Price Is Higher Than $B$’s?

No—not at all. Look at `ASSET_A1` and `ASSET_A2` in the notebook: `ASSET_A1` is around $140$ while `ASSET_A2` is around $36$ ($\log P_{A1} \approx 4.95$, $\log P_{A2} \approx 3.59$).

Why doesn’t raw price level matter?

1. **Log prices & percentage returns:** Everything is measured in log differences ($\Delta \log P$) and percentage returns (`pct_change()`), not raw dollar differences like $P_A - P_B$.

2. **Intercept & rolling mean absorb level offsets:** Any baseline difference between $\log P_A$ and $\hat{\beta} \log P_B$ sits in the regression intercept $\alpha$, and subtracting `roll_mean` centers $s_t$ around $0$ before computing `z_score`.

So in the state machine:

* **Long spread ($+1$):** Long $A$, short $\hat{\beta}$ of $B$ $\rightarrow$ profits whenever $r_A - \hat{\beta} r_B > 0$ ($A$ outperforms $B$ relative to $\hat{\beta}$).

* **Short spread ($-1$):** Short $A$, long $\hat{\beta}$ of $B$ $\rightarrow$ profits whenever $r_A - \hat{\beta} r_B < 0$ ($A$ underperforms $B$ relative to $\hat{\beta}$).

Regardless of whether $P_A > P_B$ or $P_A < P_B$, the signal is purely about whether the spread $s_t$ is above or below its recent rolling average.