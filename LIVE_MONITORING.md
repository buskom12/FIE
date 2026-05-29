# Live Monitoring Cheat-Sheet (First Hours)

## 1) Core metrics to watch
- Equity curve: steady growth or controlled pullbacks.
- Drawdown: avoid sudden jumps above baseline (~0.35).
- Rolling Sharpe: watch for stability, avoid sharp drops.
- Trades / PnL: expected trade frequency, no unexpected PnL degradation.

## 2) Alert conditions
- Max DD: `> 0.40` -> inspect tail-D triggers and consider temporary lower `FIE_D_TAIL_DD`.
- Size: `> 0.25` or `< 0.02` -> inspect Kelly/variance and `kelly_k`.
- Kelly fraction: `> 10.0` (abs) -> inspect variance floor/smoothing.
- Equity drop in 1h: `> 5%` -> inspect trades, then PnL vs edge/size/Kelly.

## 3) Real-time diagnostics
- PnL vs Kelly fraction: expect positive slope.
- Size vs PnL: larger size should be profitable on average.
- Variance vs Size: expect inverse relationship (variance up -> size down).
- Tail-D hits: monitor activation frequency vs expected regime.

## 4) Runtime alert config (`services/prod_loop.py`)
- `FIE_ALERTS_ENABLED=1`
- `FIE_ALERT_MAX_DD=0.40`
- `FIE_ALERT_MIN_SIZE=0.02`
- `FIE_ALERT_MAX_SIZE=0.25`
- `FIE_ALERT_MAX_KELLY=10.0`
- `FIE_ALERT_EQUITY_DROP_1H=0.05`
- `FIE_ALERT_COOLDOWN_SEC=300`

Example:

```bash
FIE_FAKE_SIGNALS=0 \
FIE_POLL_SECONDS=60 \
FIE_ALERTS_ENABLED=1 \
FIE_ALERT_MAX_DD=0.40 \
python3 services/prod_loop.py
```
