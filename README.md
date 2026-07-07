# Polymarket BTC 15m bot — ChainVector Edition

Autonomous trading daemon for Polymarket's 15-minute "Bitcoin Up or Down"
binary markets (`btc-updown-15m-{unix_window_start}`). It wakes for each
15-minute window, runs a Markov signal stack over tick-derived 1-minute
candles, blends in the ChainVector Probability Engine, passes a deep stack
of entry vetoes, and places real CLOB limit orders with stop-loss /
take-profit / reversal-exit management.

Execution runs through `polymarket_client.py` — Gamma API discovery, CLOB V2
order placement (EIP-712 signed via `py-clob-client-v2`), and Data-API
positions — while the battle-tested daemon logic stays in the legacy
exchange schema it was tuned on (YES/NO integer-cent quotes: YES = "Up",
NO = "Down").

**[ChainVector](https://chainvector.com) is the sole derived-signals
provider.** Polymarket is used only as the execution venue (plus its own
orderbook/trade tape and its `crypto-price` endpoint — the "price to beat"
strike and the resolution source), and Coinbase/OKX/Deribit index reads
survive only as the raw spot-price fallback chain. Every other market
signal — probabilities, momentum, liquidations, order flow, funding,
whales, quote stability, regime — comes from the ChainVector API.

## Signal architecture

| Signal | ChainVector endpoint | Role |
|---|---|---|
| Terminal probability | `/probability` (target=strike, `close_ts`=exact market close) | **Weighted EV signal** — six-estimator ensemble blended into `combined_p` via `--ev-tp-weight`, plus a hard floor veto (`--cv-prob-veto-max`) |
| Futures momentum scorecard | `/momentum` | **EV weight when it agrees** (capped tanh nudge of `combined_p`), **veto when strongly against** (signed aggregate ≤ −65 with breadth confirmation) |
| Futures lead-lag feed | `/momentum` (binance_futures + okx venues) | 6-second lead veto, dual-venue consensus veto, OKX boost, fast-exit oracle — one background poller feeds all of them |
| Prediction bid-price stability | `/predictions/stability` | Entry veto when the Polymarket quote is repricing hard against the side (cross-checks the local bid-stability gate) |
| Prediction model edge | `/predictions/edge` | Recorded signal (logged per fire, promotable after validation) |
| Prediction markets/quotes/trades/results | `/predictions/*` | Discovery, research and backtest joins |
| Liquidation cascade risk | `/liquidations/cascade-risk` | Entry veto when cascade risk ≥ 75 and the at-risk side's forced flow points against the position; 5m liquidation deltas feed the flip-prob composite |
| Liquidation heatmap | `/liquidations/heatmap` | Magnet/skew features (recorded + flip-prob composite input) |
| Combined snapshot | `/signals/snapshot` | Book imbalance, whale pressure, funding, OI, cascade — one call bundles the signal_feeds inputs |
| Taker flow / CVD | `/orderflow/cvd` | Perp taker-flow feature (recorded) |
| Long/short positioning | `/long-short`, `/positioning` | Contrarian crowd read (recorded) |
| Volatility / regime / risk index | `/volatility`, `/regime`, `/risk-index` | Recorded context bundle per fire |
| 1m candles | `/candles` | The Markov stack's bar source (aggregated client-side to 5m/15m) |

All ChainVector reads are **fail-open**: if the API is down or the key is
missing, the affected gates go neutral and the daemon falls back to its core
Markov + Polymarket-native logic.

### ChainVector state collector (`cv_collector.py`)

A second layer of CV gates reads a local JSON state file instead of hitting
the API inside the decision path. `cv_collector.py` polls ChainVector every
~5s and writes `cv_state/latest_{ASSET}.json` (probability at the current
window's strike with exact TTE, momentum scorecard, signals snapshot,
cascade risk, `/predictions/edge` for the live market, volatility, and a
rolling probability window for whipsaw/spike detection). The daemon reads
it read-only and every consumer fails open when the file is missing or
stale — so the collector can die without ever blocking the daemon.

```bash
# terminal 1 — collector
python cv_collector.py --asset BTC --state-dir cv_state

# terminal 2 — daemon with the collector-backed gates armed
python trade_daemon.py --cv-asset BTC --cv-state-dir cv_state \
    --cv-shadow-enabled --cv-flow-veto-enabled --cv-ev-veto-enabled \
    --cv-rev-veto-enabled --cv-rev-exit-enabled --cv-sl-defer-enabled
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env.local
# edit .env.local:
#   POLYMARKET_PRIVATE_KEY    — wallet key that signs CLOB orders
#   POLYMARKET_PROXY_ADDRESS  — deposit-wallet/funder address (website accounts)
#   CHAINVECTOR_API_KEY       — from https://chainvector.com (cv_live_...)
```

## Run

Defaults are baked in — running with no flags is equivalent to the full
canonical production command (bankroll $500, EV gate, strict high-conv
filters, $75 dollar-stops, resting take-profit at +10¢, hold-to-win on
golden, and the full veto stack):

```bash
cd python-service
python trade_daemon.py            # live trading
python trade_daemon.py --dry-run  # simulate only, no real orders
```

`python trade_daemon.py --help` lists every gate and threshold; each
baked-in default has a matching `--no-*` inverse flag. The full reference
is below.

# To run using ChainVector's Compute VM use this start command:
python python-service/trade_daemon.py 

## Strategy templates (`--strategy`)

Named presets over the ChainVector gate stack. A preset only touches knobs
you did **not** pass explicitly, so any flag can still override its preset
value (e.g. `--strategy momentum --cv-mom-boost-weight 0.02`).

| Template | Thesis | What it changes vs baseline |
|---|---|---|
| `baseline` (default) | Canonical production run | Nothing — full veto stack + production thresholds. |
| `conservative` | Fewer, better-confirmed entries | Prob-engine floor 0.22→0.35, momentum against-veto 65→50 (breadth 0.60→0.50), stability veto 45→30, cascade veto 75→60, 6s futures-lead veto 5→3 bps, strike cushion 0.10→0.15%, dollar-stops $75→$50. |
| `aggressive` | More entries, higher variance | Prob floor 0.22→0.15, momentum veto 65→80, stability veto 45→60, cascade veto 75→85, lead veto 5→8 bps, no cushion floor, TP-relaxed Markov gap. |
| `momentum` | Trade with the cross-venue futures tape | `/momentum` EV boost ±5pp→±10pp saturating at agg ±35, against-veto fires at 45 (breadth 0.50), OKX boost ±6pp→±8pp, lead veto 4 bps. |
| `probability` | Let the six-estimator ensemble lead | `--ev-tp-weight` 0.5→0.7 (prob engine dominates `combined_p`), prob floor 0.35, TP-relaxed gap floor. |
| `stability` | Only enter markets pricing you in | `/predictions/stability` veto at 25, bid-stability lookback 60→90s, max fade 4→2¢, burst confirm 3 samples, cushion 0.12%. |

```bash
python trade_daemon.py --strategy conservative --dry-run
python trade_daemon.py --strategy momentum --bankroll 1000
```

The applied preset (and every knob it set) is logged at startup as
`[STRATEGY] '<name>' preset applied: ...`.

## Command-line reference

All flags are optional — the defaults ARE the canonical production
baseline. Boolean features list their state (**on**/**off**) plus the flag
that flips them. "Tier" refers to the entry tiers: `golden` (65–73¢ band),
`standard`, `strong` (strong-floor), `high_conv`, `late_sure`, `late_dir`.

### Core

| Flag | Default | Description |
|---|---|---|
| `--dry-run` | off | Simulate trades; no real orders are placed. |
| `--bankroll` | `500` | Starting bankroll in USD (replaced by the live Polymarket USDC balance at startup when available). |
| `--strategy` | `baseline` | Strategy template: `baseline` \| `conservative` \| `aggressive` \| `momentum` \| `probability` \| `stability` (see the table above). Explicit flags always win over the preset. |

### ChainVector probability engine (terminal probability)

| Flag | Default | Description |
|---|---|---|
| `--no-term-prob` | engine **on** | Disable the ChainVector terminal-probability engine entirely. |
| `--term-prob-relax` | off | Allow TermProb-confirmed signals to use a relaxed Markov gap floor (0.08). |
| `--ev-tp-weight` | `0.5` | Weight of the probability engine vs Markov in `combined_p` — this is how the engine acts as the weighted EV signal. |
| `--no-cv-prob-veto` | veto **on** | Disable the probability-engine floor veto. |
| `--cv-prob-veto-max` | `0.22` | Hard floor: block entry when the six-estimator ensemble gives our side ≤ this at the exact time-to-close. |

### ChainVector momentum scorecard & futures lead-lag

| Flag | Default | Description |
|---|---|---|
| `--no-cv-mom-boost` | boost **on** | Disable the momentum-scorecard EV boost (capped tanh nudge of `combined_p` toward the cross-venue aggregate). |
| `--cv-mom-boost-weight` | `0.05` | Max `combined_p` nudge from the scorecard (±5pp). |
| `--cv-mom-boost-scale` | `50` | Aggregate score (of ±100) at which the boost saturates. |
| `--no-cv-mom-veto` | veto **on** | Disable the momentum strong-against veto. |
| `--cv-mom-veto-score` | `65` | Signed aggregate score at/below −this blocks entry ("strongly against"). |
| `--cv-mom-veto-breadth` | `0.60` | Fraction of venues that must be moving the adverse way to confirm the veto. |
| `--no-futures-lead` | feed **on** | Disable the futures lead-lag veto (binance_futures venue of `/momentum`). |
| `--futures-lead-lookback` | `6` | Seconds of lead-venue move inspected before entry. |
| `--futures-lead-veto-bps` | `5` | Veto entry if futures moved more than this many bps against the direction in the lookback. |
| `--no-okx` | view **on** | Disable the OKX venue view (consensus-veto partner + boost source). |
| `--okx-poll-interval-s` | `3.0` | `/momentum` poll cadence for the lead feed (3s = 20 req/min). |
| `--okx-boost-enabled` / `--no-okx-boost` | **on** | Nudge `combined_p` toward the OKX 6s futures move. |
| `--okx-boost-weight` | `0.06` | Max `combined_p` nudge from the OKX boost (±6pp). |
| `--okx-boost-scale` | `0.015` | OKX 6s move (%) at which the boost saturates. |

### ChainVector prediction-market & liquidation gates

| Flag | Default | Description |
|---|---|---|
| `--no-cv-stab-veto` | veto **on** | Disable the `/predictions/stability` quote-stability veto (blocks while the Polymarket quote is repricing hard against our side). |
| `--cv-stab-mom-against` | `45` | Signed stability momentum_score (±100 scale) at/below −this blocks entry. |
| `--no-cv-cascade-veto` | veto **on** | Disable the liquidation cascade-risk veto. |
| `--cv-cascade-veto-score` | `75` | Cascade risk_score at/above which the veto can fire (requires the at-risk side's forced flow to point against the position). |

### Cross-venue consensus veto (binance_futures + OKX)

| Flag | Default | Description |
|---|---|---|
| `--no-consensus-veto` | veto **on** | Disable the dual-venue consensus veto (both venues showing a directional move against the trade). |
| `--consensus-min-move-pct` | `0.005` | Min \|move %\| for a venue to count as directional (below = neutral). |
| `--consensus-okx-lookback-s` | `6.0` | OKX lookback window (seconds) for the consensus check. |
| `--no-consensus-smart-bypass` | bypass **on** | Disable the three noise-filters that skip the veto when the 6s blip contradicts the broader trend. |
| `--consensus-long-window-s` | `60` | Bypass 1: longer futures window checked against the 6s blip. |
| `--consensus-long-min-pct` | `0.030` | Bypass 1: min favorable \|move %\| in the longer window. |
| `--consensus-5m-favor-pct` | `0.10` | Bypass 2: min favorable sum of recent 5m bars (%). |
| `--consensus-far-dist-pct` | `0.25` | Bypass 3: min \|dist from strike %\| for tiny moves to be ignored. |
| `--consensus-far-max-move-pct` | `0.020` | Bypass 3: max venue move (%) still considered "tiny". |

### EV gate & strong-floor

| Flag | Default | Description |
|---|---|---|
| `--ev-gate` / `--no-ev-gate` | **on** | Replace flat YES/NO price caps with an EV-based override when the price cap is the only blocker. |
| `--ev-floor` | `0.05` | Min EV per contract ($) for the standard EV override. |
| `--ev-ceiling` | `90` | Hard price ceiling (¢) regardless of EV. |
| `--ev-strong-floor` | `0.0` | Relaxed EV floor ($) when strong-signal criteria hold. |
| `--ev-strong-gap-min` | `0.13` | Min Markov gap to qualify for strong-floor. |
| `--ev-strong-price-max` | `88` | Max entry price (¢) for strong-floor. |
| `--ev-strong-tp-min` | `0.55` | Min directional TP confirmation for strong-floor (YES ≥ 0.55, NO ≤ 0.45). |
| `--ev-strong-max-mins` | `8.0` | Max minutes-left for strong-floor entries. |
| `--ev-strong-max-adverse-momentum` | `0.10` | Max adverse cumulative 5m momentum (%) allowed for strong-floor. |
| `--strong-floor-min-stake-pct` | `0.020` | Floor stake (fraction of bankroll) when Kelly returns 0 on a strong-floor approval. |
| `--no-strong-floor-hurst-bypass` | bypass **on** | Stop strong-floor from bypassing the mean-reverting Hurst gate. |
| `--ev-walkup-override-enabled` / `--no-ev-walkup-override` | **on** | On book-confirmed "sure" trades, relax the walk floor so the price walk can win fills. |
| `--ev-override-pwin-min` | `0.70` | Min p_win for the walk-up override. |
| `--ev-override-price-max` | `85` | Max price (¢) the override will walk to. |
| `--ev-override-book-skew-min` | `0.0` | Min signed book_skew required (book must support the trade). |
| `--ev-override-floor` | `-0.10` | Relaxed EV floor ($/contract) used during the override walk. |

### Entry tiers

| Flag | Default | Description |
|---|---|---|
| `--standard-price-cap-yes` | `88` | STANDARD tier YES price cap (¢). |
| `--standard-price-cap-no` | `88` | STANDARD tier NO price cap (¢). |
| `--standard-min-entry-contracts` | `2` | Skip STANDARD entries Kelly-sized below this (window stays open for a better entry). |
| `--golden-price-lo` / `--golden-price-hi` | `65` / `73` | Golden-zone price band (¢). |
| `--golden-no-dist` | off | Drop the near-strike distance gate for golden entries. |
| `--golden-no-hurst` | off | Drop the Hurst gate for golden entries. |
| `--high-conv-gap-min` | `0.40` | Min Markov gap for HIGH-CONV (strict baseline). |
| `--high-conv-persist-min` | `0.95` | Min Markov persistence for HIGH-CONV. |
| `--high-conv-tp-strong` | `0.90` | TP confirmation threshold for HIGH-CONV (YES ≥ 0.90 / NO ≤ 0.10) — reachable because the ChainVector engine prices exact TTE. |
| `--high-conv-price-max` | `97` | Max bid (¢) for HIGH-CONV. |
| `--high-conv-ev-floor` | `0.005` | EV floor ($/contract) for HIGH-CONV (positive-EV only). |
| `--high-conv-max-mins` | `13.0` | Max minutes-left for HIGH-CONV (timing bypass on extreme signals). |
| `--high-conv-dist-min` | `0.25` | Min \|dist from strike %\| for HIGH-CONV (0 disables). |
| `--no-high-conv-vol-bypass` | bypass **on** | Stop HIGH-CONV from bypassing the vol gate on directional moves. |
| `--high-conv-vol-bypass-momentum` | `0.10` | Min 5m return (%) in trade direction for the vol bypass. |
| `--high-conv-vol-bypass-distance` | `0.15` | Min \|dist %\| for the vol bypass. |
| `--high-conv-vol-bypass-strong-distance` | `0.25` | \|dist %\| at which distance alone qualifies (momentum check skipped). |
| `--late-window-mins` | `0.0` | Minutes-left threshold for the LATE-SURE tier — **0 = tier disabled (baseline)**. |
| `--late-window-price-max` | `98` | Max bid (¢) for LATE-SURE. |
| `--late-window-min-tp` | `0.75` | TP threshold confirming direction for LATE-SURE. |
| `--late-window-ev-floor` | `-0.10` | EV floor ($/contract) for LATE-SURE. |
| `--late-sure-min-stake-pct` | `0.015` | Floor stake for LATE-SURE when Kelly = 0. |
| `--no-late-sure-vol-bypass` | bypass **on** | Stop LATE-SURE from bypassing the high-vol gate. |
| `--no-late-dir` | tier **on** | Disable the LATE-DIR tier (late-window directional vol bypass). |
| `--late-dir-mins` | `5.0` | Max minutes-left for LATE-DIR. |
| `--late-dir-gap-min` | `0.25` | Min Markov gap for LATE-DIR. |
| `--late-dir-persist-min` | `0.95` | Min Markov persistence for LATE-DIR. |
| `--late-dir-distance-min` | `0.05` | Min \|dist %\| for LATE-DIR. |
| `--late-dir-momentum-min` | `0.05` | Min 5m return (%) in trade direction (skipped at strong distance). |
| `--late-dir-strong-distance` | `0.08` | \|dist %\| at which the momentum check is skipped. |
| `--late-dir-ev-floor` | `-0.05` | EV floor ($/contract) for LATE-DIR. |
| `--late-dir-price-max` | `89` | Max bid (¢) for LATE-DIR. |

### Orderbook lock-in bypasses

| Flag | Default | Description |
|---|---|---|
| `--no-orderbook-lockin` | **on** | Disable lock-in detection (tight spread + deep top-of-book + Markov gap ⇒ HC bypasses its TP threshold, LATE-SURE cap rises to 99¢). |
| `--orderbook-lockin-spread-max` | `2` | Max spread (¢) to qualify as lock-in. |
| `--orderbook-lockin-price-min` | `85` | Min top-of-book bid (¢) in the trade direction. |
| `--orderbook-lockin-gap-min` | `0.20` | Min Markov gap combined with the book for lock-in. |
| `--late-window-price-max-lockin` | `99` | LATE-SURE price cap (¢) when lock-in is confirmed. |
| `--hc-lockin-ev-floor` | `-0.10` | HIGH-CONV EV floor ($/contract) under lock-in. |
| `--hc-lockin-min-stake-pct` | `0.015` | Floor stake for lock-in HC trades when Kelly = 0. |

### Entry vetoes (Markov/price-action)

| Flag | Default | Description |
|---|---|---|
| `--hurst-tp-veto-enabled` / `--no-hurst-tp-veto` | **on** | Block when Hurst is high (trending) AND the probability engine disagrees with Markov in the adverse direction ("chasing a fading top"). |
| `--hurst-tp-veto-min-hurst` | `0.80` | Min Hurst to arm the veto. |
| `--hurst-tp-veto-min-diff` | `0.05` | Min adverse \|TP − Markov\| to fire. |
| `--hurst-tp-veto-tiers` | `high_conv,strong` | Tiers the veto applies to. |
| `--max-adverse-bar-veto-enabled` / `--no-max-adverse-bar-veto` | **on** | Block direction X when any recent 5m bar moved ≥ threshold against X (falling-knife / dead-cat-bounce guard). |
| `--max-adverse-bar-veto-pct` | `0.30` | Adverse single-bar threshold (%). |
| `--max-adverse-bar-veto-tiers` | `high_conv,strong` | Tiers the veto applies to. |
| `--no-cum-adverse-momentum-veto` | veto **on** | Disable the cumulative adverse-drift veto (sum of recent 5m bars against the trade). |
| `--cum-adverse-momentum-veto-pct` | `0.35` | Cumulative adverse threshold (%). |
| `--cum-adverse-momentum-veto-tiers` | `high_conv,strong,late_sure` | Tiers the veto applies to. |
| `--no-hc-low-hurst-veto` / `--hc-low-hurst-veto` | **off** (baseline) | LOW-HURST HC veto: block extreme-Markov HC trades in mean-reverting regimes. |
| `--hc-low-hurst-threshold` | `0.30` | Hurst below this = mean-reverting regime. |
| `--hc-low-hurst-markov-extremity` | `0.35` | Min \|Markov − 0.5\| for the veto to apply. |
| `--last-bar-adverse-threshold` | `0.10` | Universal gate: block if the most recent 5m bar moved against the trade by more than this (%). 0 disables. |
| `--min-entry-cushion-pct` | `0.10` | Skip entries with \|BTC − strike\|/strike below this (%) — near-money entries are net-losing. 0 disables. |
| `--min-entry-price` | `0` | Hard entry-price floor (¢), all tiers. 0 = off. |

### Entry vetoes (market microstructure)

| Flag | Default | Description |
|---|---|---|
| `--taker-flow-veto-enabled` / `--no-taker-flow-veto` | **on** | Block when recent Polymarket taker aggression is heavily against the bet near the strike. |
| `--taker-flow-veto-agg-min` | `0.90` | Fraction of taker volume against the bet to trigger. |
| `--taker-flow-veto-dist-max` | `0.25` | Only applies within this \|dist %\| of the strike. |
| `--taker-flow-veto-min-trades` | `5` | Min recent trades before the veto can fire. |
| `--bid-stab-veto-enabled` / `--no-bid-stab-veto` | **on** | Require our side's Polymarket bid to be stable/rising over the lookback (cross-checked by the ChainVector stability veto). |
| `--bid-stab-lookback-s` | `60` | Bid-stability lookback (seconds). |
| `--bid-stab-max-fade-cents` | `4` | Max cents the bid may sit below its lookback peak. |
| `--bid-stab-min-samples` | `3` | Min samples before the gate arms (fewer = fail-open). |
| `--bid-stab-burst-samples` | `3` | Live re-samples at the entry moment; any downtick vetoes. 0 disables. |
| `--bid-stab-burst-interval-s` | `2.0` | Seconds between burst re-samples. |
| `--perp-veto-enabled` / `--no-perp-veto` | **on** | Skip entry when the perp 30s tape moves against the position. |
| `--perp-veto-m30s-threshold` | `-10.0` | Signed 30s perp momentum (bp toward the trade) at/below which entry is vetoed. |
| `--book-skew-veto-enabled` | **off** | Skip golden entries when futures resting-depth skew is stacked against the trade (log-only verdicts still recorded). |
| `--book-skew-threshold` | `-0.15` | Signed book_skew at/below which a golden entry is vetoed. |
| `--book-skew-all-tiers` | off | Apply the book-skew veto to all tiers (not recommended). |
| `--perp-imb-veto` | **off** | Skip entries when the perp book imbalance is deeply against the position. |
| `--perp-imb-veto-threshold` | `-0.50` | Signed imbalance at/below which entry is vetoed. |
| `--golden-near-vol-veto` / `--no-golden-near-vol-veto` | **on** | Skip golden entries that are near-strike AND in elevated short-term vol (knife-edge bucket). |
| `--golden-near-vol-dist-max` | `0.08` | \|dist %\| below which a golden entry counts as "near". |
| `--golden-near-vol-gk-min` | `0.0020` | GK vol at/above which it counts as "high-vol". |
| `--hurst-cushion-veto-enabled` | off | Block near-strike entries in a mean-reverting regime (Hurst < max AND \|dist\| < min — the near-strike coin-flip bucket; BTC 30d backtest NET +$2,163). |
| `--hurst-cushion-hurst-max` | `0.40` | Hurst below this = mean-reverting. |
| `--hurst-cushion-dist-min` | `0.20` | \|dist %\| below this = near-strike. |
| `--chase-veto-enabled` | off | Block entries whose ask walked ≥ walkup-cents above its lookback low while Hurst < max (buying the re-test top of a fading move). Native — no ChainVector dependency. |
| `--chase-veto-walkup-cents` | `22` | Min ask walk-up (¢) off the lookback low. |
| `--chase-veto-hurst-max` | `0.45` | Hurst must be below this (mean-reverting) for the veto to arm. |
| `--chase-veto-lookback-mins` | `4.0` | Ask-history lookback (minutes). |

An always-on **adverse-M60S high-price veto** (no flag) also blocks entries
priced ≥ 90¢ while the 60s perp tape runs ≥ 10bp against the position —
paying near-max price to fight an in-progress move (fleet backtest NET +$660).

### ChainVector collector gates (require `cv_collector.py` + `--cv-asset`)

All of these read `latest_{asset}.json` and **fail open** when the file is
missing/stale. `market_id` guards mean the probability-based gates only act
when the collector priced the exact market the daemon is trading.

| Flag | Default | Description |
|---|---|---|
| `--cv-shadow-enabled` | off | Stamp the full ChainVector feature set into every audit record (record-only; needs `--cv-asset`). |
| `--cv-asset` | `""` | ChainVector asset symbol (e.g. `BTC`). Setting it resolves `cv_state_path` for every collector-backed gate. |
| `--cv-state-dir` | `cv_state` | Directory the collector writes to (or `CV_STATE_DIR` env). |
| `--cv-max-age-s` | `20` | Max state-file age for the shadow stamp. |
| `--cv-flow-veto-enabled` | off | Block near-strike entries when cross-venue breadth/momentum is against the side, or a liquidation cascade is running against it. |
| `--cv-flow-breadth-min` | `0.25` | Breadth-for-side below this (near strike) blocks. |
| `--cv-flow-mom-min` | `-25.0` | Signed momentum-for-side at/below this (near strike) blocks. |
| `--cv-flow-cascade-min` | `60.0` | Cascade risk against the side at/above this blocks at any distance. |
| `--cv-flow-dist-max` | `0.10` | \|dist %\| below which the breadth/momentum arms apply. |
| `--cv-flow-max-age-s` | `20` | Max state age (seconds). |
| `--cv-ev-veto-enabled` | off | Block when the CV ensemble p-for-side < prob-min, or \|dist\| < cushion-sigma-min × the CV expected-move sigma at exact TTE (calibration: Brier 0.06–0.10 vs 0.19 for the venue mid). |
| `--cv-ev-prob-min` | `0.75` | Ensemble p-for-side floor. |
| `--cv-ev-cushion-sigma-min` | `0.50` | Min cushion in expected-move sigmas. |
| `--cv-ev-max-age-s` | `30` | Max state age (seconds). |
| `--cv-rev-veto-enabled` | off | Flip-market detector: block when p-for-side fell ≥ drop-min from its window peak (< p-max), or a "lone pump" is running (whale flow against + no breadth + momentum against). |
| `--cv-rev-drop-min` | `0.12` | Min p_for drop from the window peak. |
| `--cv-rev-p-max` | `0.90` | Drop arm only fires below this p_for. |
| `--cv-rev-whale-max` | `-60.0` | Lone-pump: whale-flow-for-side at/below this. |
| `--cv-rev-breadth-max` | `0.25` | Lone-pump: breadth-for-side at/below this. |
| `--cv-rev-mom-max` | `0.0` | Lone-pump: momentum-for-side at/below this. |
| `--cv-rev-max-age-s` | `30` | Max state age (seconds). |
| `--cv-rev-cross-min` | `0` | Whipsaw arm: veto when spot crossed the strike ≥ this many times in 4m. 0 = off. |
| `--cv-rev-spike-min` | `0.0` | Fresh-spike arm: veto when p_for rose ≥ this within 3m (unconsolidated move). 0 = off. |
| `--cv-edge-veto-shadow` | off | Shadow-log entries where the `/edge` model prices our side below the market (never blocks). |
| `--cv-edge-veto-max` | `-0.05` | Shadow fires when model_for − market_for ≤ this. |

### ChainVector collector exits & stop-loss holds

| Flag | Default | Description |
|---|---|---|
| `--cv-rev-exit-enabled` | off | In-trade exit when the CV ensemble p-for-side falls ≥ exit-drop-min from its in-trade peak to < exit-p-max over ≥ `--cv-rev-exit-streak` collector rows (tape replay: caught 22/23 reversals ~4 min early, 11% false fires). Skips high_conv/late_sure; one live fire per position; shadow arm logs the aggressive 0.12/0.90 variant. |
| `--cv-rev-exit-drop-min` | `0.20` | Min p_for drop from the in-trade peak. |
| `--cv-rev-exit-p-max` | `0.80` | Fires only below this p_for. |
| `--cv-rev-exit-min-bid` | `40` | Min bid (¢) — collapsed bids belong to the $-stop. |
| `--cv-rev-exit-streak` | `2` | Consecutive collector rows required for a live fire. |
| `--cv-rev-exit-lm-gate` | off | Only fire once CV spot is on the losing side of the strike (or p_for < lm-floor escape hatch). |
| `--cv-rev-exit-lm-floor` | `0.45` | p_for below this fires regardless of spot side. |
| `--cv-sl-defer-enabled` | off | Defer the cents stop-loss while CV spot is on our side of the strike AND p_for ≥ p-min (5-week study: 35% of stops clipped winners; deferral returned ~+$1.9k). Dollar-cap fires are never deferred. |
| `--cv-sl-defer-p-min` | `0.75` | Min ensemble p_for to defer. |

An always-on **underlying-aware SL hold** (no flag) additionally skips
book-driven stop triggers while the basis-free perp spot estimate is still
on our side of the strike by ≥ 0.02% — fleet backtest: 5/11 historical SL
exits were book fake-outs (~+$488 if held). RRM/pcross/CV-REV exits bypass
the hold (already underlying-gated).

### Sizing & confirmation boosts

| Flag | Default | Description |
|---|---|---|
| `--standard-confirmed-boost` | `1.5` | Kelly multiplier for non-HC tiers when both lead venues confirm direction. 1.0 disables. |
| `--high-conv-confirmed-frac` | `0.10` | Kelly fraction for HIGH-CONV when both venues confirm (2× the 0.05 baseline). |
| `--no-hc-block-on-split` | block **on** | Disable the HIGH-CONV split-externals block (one venue opposes, the other doesn't ⇒ HC demoted). |
| `--max-trade-usd` | `800` | Hard ceiling on a single ticket's capital (contracts × entry). 0 = uncapped. |

### Order execution

| Flag | Default | Description |
|---|---|---|
| `--order-lead-cents` | `2` | Submit the initial order this many ¢ above the observed ask (sweeps fast-moving books; EV-checked). |
| `--retry-walk-cents` | `8` | On zero-fill IOC, walk the price up to N¢; each step re-checks EV. |
| `--retry-plus-cent` | off | Back-compat alias for `--retry-walk-cents 1`. |
| `--max-window-fill-attempts` | `10` | Max IOC retry attempts per window (signal re-evaluated each attempt). |
| `--refill-retry-sleep-s` | `10` | Seconds between retry attempts within a window. |
| `--patient-topup-enabled` / `--no-patient-topup` | **on** | If a position filled below intent, keep watching and top up when a fresh signal approves at an improved price. |
| `--patient-topup-interval-s` | `20` | Min seconds between top-up attempts. |
| `--patient-topup-min-mins` | `2.5` | Stop top-ups below this many minutes left. |
| `--patient-topup-dynamic-kelly` | off | Let top-ups grow toward the fresh Kelly size instead of the frozen entry-time intent. |

### Stop-loss & fast exits

| Flag | Default | Description |
|---|---|---|
| `--no-sl` | SL **on** | Disable the stop-loss monitor entirely. |
| `--sl-loss-cents` | `99` | Per-contract stop (¢ below entry) for golden/standard/strong/late-dir. 99 = effectively off; the dollar stop is the live protection. |
| `--sl-loss-cents-high-conv` | `99` | Per-contract stop for HIGH-CONV + LATE-SURE. 99 = effectively off. |
| `--max-loss-per-trade` | `75` | Hard dollar stop for golden/standard positions. 0 disables. |
| `--max-loss-per-trade-high-conv` | `75` | Hard dollar stop for HIGH-CONV positions. 0 disables. |
| `--sl-grace-mins` | `1.5` | Minutes after entry before the SL can fire. |
| `--sl-disable-late-mins` | `1.5` | Disable the SL below this many minutes left (settlement handles it). |
| `--sl-poll-interval-s` | `2.0` | SL poll cadence for standard-price tiers. |
| `--sl-poll-interval-hc-s` | `2.0` | SL poll cadence for high-price tiers. |
| `--sl-trigger-mode` | `mid` | Trigger price: `bid` (conservative), `mid` (faster), or `last`. Selling always hits the bid. |
| `--no-sl-aggressive-sell` | aggressive **on** | Disable extreme-limit IOC sells on SL exits (aggressive fills against any remaining bid). |
| `--no-futures-fast-exit` / `--futures-fast-exit` | **off** (baseline) | Futures fast-exit: during SL monitoring, exit immediately if the lead-venue futures moved sharply against the position. |
| `--futures-fast-exit-window-s` | `30` | Fast-exit lookback (seconds). |
| `--futures-fast-exit-threshold-pct` | `0.20` | Adverse futures move (%) that triggers the fast exit. |
| `--futures-fast-exit-sanity-max-pct` | `5.0` | Any \|move\| above this is treated as a bad tick and ignored. |

### Take-profit & hold-to-win

| Flag | Default | Description |
|---|---|---|
| `--take-profit-enabled` / `--no-take-profit` | **on** | Exit at entry + `--take-profit-cents`. |
| `--take-profit-cents` | `10` | Profit target (¢ above entry). |
| `--take-profit-all-trades` / `--take-profit-perp-confirmed-only` | **all trades** | Apply the TP to every trade vs only perp-momentum-confirmed entries. |
| `--take-profit-perp-min` | `0.0` | Min entry perp 30s momentum (bp) to arm the TP in perp-confirmed mode. |
| `--resting-tp-enabled` / `--no-resting-tp` | **on** | Place a real resting GTC limit sell at entry+TP on fill (catches spikes between polls; cancelled before other exits and near settlement). |
| `--high-price-tp-enabled` | off | Ceiling TP: entries ≥ min-cents rest a sell at target-cents instead of entry+TP. |
| `--high-price-tp-min-cents` | `89` | Entry price (¢) at/above which the ceiling TP applies. |
| `--high-price-tp-target-cents` | `98` | Resting sell price (¢) for the ceiling TP. |
| `--highrisk-tp-enabled` / `--no-highrisk-tp` | **on** | Conditional tight TP for high_conv trades showing the last-minute-reversal signature (drawdown + near-strike entry). |
| `--highrisk-tp-dd-cents` | `12` | Min drawdown (¢ below entry) to flag high-risk. |
| `--highrisk-tp-dist-max` | `0.35` | Max \|entry dist %\| for the high-risk gate. |
| `--highrisk-tp-cents` | `5` | Tight TP offset (¢ above entry). |
| `--holdwin-enabled` / `--no-holdwin` | **on** | Hold-to-win: on clearly-working winners, cancel the resting TP and ride to settlement; re-arms if conviction deteriorates. |
| `--holdwin-tiers` | `golden` | Tiers eligible for hold-to-win. |
| `--holdwin-min-profit-cents` | `3` | Only cancel the TP once in profit by ≥ this (¢). |
| `--holdwin-min-dist-pct` | `0.10` | Only cancel the TP with live cushion ≥ this (%). |
| `--holdwin-max-potm` | `0.20` | Only cancel the TP when drift-aware P(end OTM) ≤ this. |
| `--holdwin-rearm-dist-pct` | `0.05` | Re-arm the TP if the cushion falls below this (%). |
| `--holdwin-rearm-potm` | `1.01` | Re-arm the TP if P(OTM) rises to this (1.01 = path effectively off; the trail re-arm still protects). |
| `--holdwin-trail-cents` | `8` | Re-arm the TP if the bid retraces this many ¢ from its peak while holding. |
| `--holdwin-min-gap` | `0.30` | Min entry Markov gap to hold a winner to settlement. |
| `--tp-reentry-enabled` | off | After a resting-TP fully sells, re-open the window for a fresh evaluation (may re-enter or flip). |
| `--sl-reentry-enabled` | off | After an adverse exit fully closes a position, re-open the window for a fresh evaluation. |

### Reversal exits (RRM & predict-cross)

| Flag | Default | Description |
|---|---|---|
| `--rrm-exit-enabled` / `--no-rrm-exit` | **on** | Sell when the reversal-risk monitor fires (strike breach + multi-signal confluence score out of 10). |
| `--rrm-exit-min-score` | `6` | Min RRM score to trigger a live exit. |
| `--rrm-exit-min-contracts` | `25` | Positions smaller than this stay log-only. |
| `--rrm-exit-cushion-max` | `0.15` | Live RRM exits only for entries within this \|dist %\| of the strike (0 = no gate). |
| `--predict-cross-exit` | off | Sell pre-emptively when drift-aware P(end OTM) exceeds the threshold for consecutive polls in the final minutes. |
| `--pcross-prob` | `0.40` | P(end OTM) threshold. |
| `--pcross-max-mins` | `2.0` | Only arms below this many minutes left. |
| `--pcross-confirm-polls` | `2` | Consecutive polls above threshold required. |
| `--pcross-min-contracts` | `25` | Positions smaller than this stay log-only. |
| `--pcross-keep-alive-mins` | `0.4` | Keep the sell path alive down to this many minutes before close. |

### Loss recovery (hedge & smart flip) — both off by default

| Flag | Default | Description |
|---|---|---|
| `--hedge-enabled` | off | After a HC/STRONG fill, buy opposite-side contracts as a downside hedge. |
| `--hedge-tiers` | `high_conv,strong` | Tiers eligible for hedging. |
| `--hedge-min-yes-entry` / `--hedge-max-yes-entry` | `70` / `88` | Primary entry price band (¢) for hedge eligibility. |
| `--hedge-max-no-cost` | `28` | Max opposite-side ask (¢) to buy the hedge at. |
| `--hedge-no-settle-assumed` | `0.95` | Conservative assumed hedge settle value ($) for sizing. |
| `--hedge-max-capital-mult` | `2.5` | Cap on total (primary + hedge) capital vs primary alone. |
| `--hedge-widened-sl-cents` | `50` | Widened SL trigger (¢) while the hedge is attached. |
| `--hedge-no-sell-target` | `97` | Sell the hedge immediately if its bid hits this (¢) after the primary SL fires. |
| `--hedge-no-sell-trail` | `10` | Sell the hedge if its bid drops this many ¢ from peak. |
| `--hedge-post-sl-poll-s` | `2.0` | Hedge-monitor poll cadence after the primary SL fires. |
| `--smart-flip-enabled` | off | After a HC/STRONG SL fires, evaluate buying the opposite side as a defensive recovery position. |
| `--smart-flip-tiers` | `high_conv,strong,late_sure` | Tiers eligible for the flip. |
| `--smart-flip-min-opp-entry` / `--smart-flip-max-opp-entry` | `50` / `75` | Opposite-side bid band (¢) at the SL moment. |
| `--smart-flip-recovery-ratio` | `0.50` | Target fraction of the primary loss to recover. |
| `--smart-flip-sl-cents` | `15` | Tight stop (¢) on the flip position. |
| `--smart-flip-sell-target` | `89` | Sell-target bid (¢) for the flip. |
| `--smart-flip-trail-cents` | `10` | Trail-stop (¢ from peak) on the flip. |
| `--smart-flip-max-capital-usd` | `100` | Hard cap on flip capital. |
| `--smart-flip-min-mins-remaining` | `5.0` | Min minutes left at the SL moment. |
| `--no-smart-flip-futures-confirm` | confirm **on** | Drop the requirement that futures confirm continuation in the flip direction. |
| `--smart-flip-futures-confirm-pct` | `0.10` | Required futures move (%) in the flip direction. |
| `--smart-flip-futures-window-s` | `30` | Futures confirmation lookback (seconds). |
| `--smart-flip-poll-s` | `2.0` | Flip-monitor poll cadence. |
| `--smart-flip-retry-attempts` | `3` | Re-checks if the first evaluation fails a gate (1 = one-and-done). |
| `--smart-flip-retry-sleep-s` | `15` | Seconds between flip retries. |

### Fade-bounce dual-entry tier — off by default

| Flag | Default | Description |
|---|---|---|
| `--fade-bounce-enabled` | off | Buy additional same-side contracts when the bid dips into the discount band mid-window. |
| `--fade-bounce-no-ask-min` / `--fade-bounce-no-ask-max` | `40` / `55` | Discount band (¢). |
| `--fade-bounce-yes-side-enabled` | off | Allow YES-side fade-bounce too. |
| `--fade-bounce-markov-no-max` | `0.45` | NO-side: cached entry Markov P(YES) must be ≤ this. |
| `--fade-bounce-markov-yes-min` | `0.55` | YES-side: cached Markov must be ≥ this. |
| `--fade-bounce-hurst-min` | `0.50` | Min cached Hurst to qualify. |
| `--fade-bounce-dist-min` | `0.03` | Min cached \|dist %\| at primary entry. |
| `--fade-bounce-min-mins` / `--fade-bounce-max-mins` | `3.0` / `12.0` | Minutes-left window for the attach. |
| `--fade-bounce-min-stake-pct` | `0.015` | Floor stake (fraction of bankroll). |
| `--fade-bounce-kelly-frac` | `0.05` | Kelly-fraction cap for the leg. |
| `--fade-bounce-sl-cents` | `20` | SL trigger (¢) for the fade-bounce leg. |
| `--fade-bounce-max-capital-usd` | `20` | Hard capital cap per attach. |

### Risk throttles — off by default

| Flag | Default | Description |
|---|---|---|
| `--rolling-wr-enabled` | off | When the rolling win rate drops below threshold, block weaker tiers until recovery or timeout. |
| `--rolling-wr-window` | `5` | Number of recent outcomes tracked. |
| `--rolling-wr-threshold` | `0.40` | WR at/below which defensive mode engages. |
| `--rolling-wr-timeout-mins` | `120` | Max time in defensive mode (0 = no timeout). |
| `--rolling-wr-defensive-tiers` | `standard,strong` | Tiers blocked while defensive. |
| `--adaptive-bankroll-enabled` | off | On drawdown/weak WR, shrink sizing to a reduced fraction instead of stopping; recovers on demonstrated WR. |
| `--adaptive-br-reduced-frac` | `0.15` | Bankroll fraction used while REDUCED. |
| `--adaptive-br-loss-trigger-usd` | `300` | Drawdown (USD from peak) that triggers REDUCED. |
| `--adaptive-br-wr-trigger` | `0.50` | Rolling WR floor that triggers REDUCED. |
| `--adaptive-br-wr-window-h` | `3.0` | Rolling WR window (hours). |
| `--adaptive-br-wr-min-trades` | `5` | Min trades in the window before it can trigger. |
| `--adaptive-br-recover-wr` | `0.75` | WR over the recovery window required to restore full sizing. |
| `--adaptive-br-recover-window` | `6` | Number of recent trades for the recovery WR. |
| `--adaptive-br-recover-min-wins` | `3` | Min wins while REDUCED before recovery is allowed. |

### Recording / observability

| Flag | Default | Description |
|---|---|---|
| `--no-orderbook-signal` | recording **on** | Disable Polymarket orderbook depth recording (audit-only). |
| `--no-trade-flow-signal` | recording **on** | Disable Polymarket recent-trades recording (taker aggression, audit-only). |
| `--trade-flow-lookback-n` | `20` | Recent Polymarket trades summarized per poll. |

## Layout

```
python-service/
  trade_daemon.py      # main daemon: gates, sizing, execution, exits
  polymarket_client.py # execution adapter: Gamma discovery, CLOB V2 orders,
                       # Data-API positions — legacy exchange schema surface
  run_backtest.py      # Markov stack + candle fetch (ChainVector 1m agg)
  chainvector.py       # ChainVector client (cached, rate-limited, fail-open)
  cv_collector.py      # ~5s ChainVector state writer (latest_{asset}.json)
                       # backing the CV-FLOW/CV-EV/CV-REV/SL-defer gates
  cv_lead.py           # /momentum background poller -> lead-lag venue views
  signal_feeds.py      # live signal snapshot + RRM + flip-prob composite
  terminal_prob.py     # probability engine wrapper (exact close_ts TTE)
  liq_heatmap.py       # heatmap feature extraction
  orderbook_feed.py    # Polymarket book/trade-tape snapshots (execution venue)
  audit_log.py         # structured JSONL audit trail
  paths.py             # non-root-safe data/log directory resolution
```

## Notes

- **Never commit `.env.local` or your wallet private key** — `.gitignore`
  covers `.env*` and key files.
- Logs, audit JSONL and caches go to the first writable of
  `$BTC15M_DATA_DIR` → package dir → `~/.polymarket_btc_15m_cv` → tmp, so the
  daemon runs cleanly as a non-root user in containers.
- Rate budget: the lead poller runs `/momentum` every 3s (20 req/min); all
  other endpoints are TTL-cached in `chainvector.py` to fit the Developer
  plan's 60 req/min with headroom.
- This trades real money on Polymarket. Verify with `--dry-run` and a small
  bankroll first.
