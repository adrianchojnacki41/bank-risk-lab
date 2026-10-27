# Bank Risk Lab

![tests](https://github.com/adrianchojnacki41/bank-risk-lab/actions/workflows/tests.yml/badge.svg)

Simulation library for credit and market risk. Computes default distributions,
expected loss, VaR and Expected Shortfall, and shows how a common economic factor
fattens the loss tail while leaving expected loss unchanged.

## Modules

- `validation.py`: input checks shared by all modules
- `credit.py`: default distribution and expected credit loss
- `fraud.py`: Bayes' theorem for fraud alarms
- `market.py`: binomial model of a stock price
- `bank_risk.py`: bank losses under economic regimes, EL, VaR and ES

## How to run

    pip install -r requirements.txt
    python -m pytest

## Model limitations

- only two economic regimes
- identical loans and constant LGD
- binomial ±1% moves do not capture fat tails of real returns
- parameters are assumed, not estimated from data
