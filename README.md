# BTC 5m EMA Mean Reversion Model

Predicts the next Coinbase five-minute candle after an extreme deviation from EMA26. The selected
3.426 ATR threshold scored 59.10% on training and 56.22% on holdout; average holdout directional
return was +0.0136% before fees, falling to +0.0015% in the latest 30 days.

```bash
python3 -m unittest discover -s tests -v
python3 ema_next_candle_research.py
```
