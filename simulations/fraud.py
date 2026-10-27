from simulations.validation import check_count, check_probability


def posterior_fraud_given_alarm(
    prevalence: float, sensitivity: float, false_positive_rate: float
) -> float:
    """Probability that a transaction is fraudulent given that it raised an alarm."""
    check_probability(prevalence, "prevalence")
    check_probability(sensitivity, "sensitivity")
    check_probability(false_positive_rate, "false_positive_rate")

    p_honest = 1 - prevalence
    p_alarm_and_honest = p_honest * false_positive_rate
    p_alarm_and_fraud = prevalence * sensitivity
    p_alarm = p_alarm_and_fraud + p_alarm_and_honest

    if p_alarm == 0:
        raise ValueError("Alarm is impossible: P(alarm) = 0")
    return p_alarm_and_fraud / p_alarm


def expected_alarm_counts(
    n_transactions: int,
    prevalence: float,
    sensitivity: float,
    false_positive_rate: float,
) -> dict[str, float]:
    """Expected counts in each cell of the fraud-alarm confusion matrix."""
    check_count(n_transactions, "n_transactions")
    check_probability(prevalence, "prevalence")
    check_probability(sensitivity, "sensitivity")
    check_probability(false_positive_rate, "false_positive_rate")

    n_fraud = prevalence * n_transactions
    n_honest = (1 - prevalence) * n_transactions

    return {
        "fraud_alarm": n_fraud * sensitivity,
        "fraud_no_alarm": n_fraud * (1 - sensitivity),
        "honest_alarm": n_honest * false_positive_rate,
        "honest_no_alarm": n_honest * (1 - false_positive_rate),
    }
