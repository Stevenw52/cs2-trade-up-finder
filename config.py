"""Runtime configuration for CS2 trade-up ROI engine."""

# Analysis budget (USD) for 10-input trade-up cost.
BUDGET_USD = 50.0

# Steam Community Market transaction fee (15%).
STEAM_TAX_PERCENT = 0.15

# Monte-Carlo samples used to estimate output float and profit variance.
MONTE_CARLO_SIMS = 5000

# Price cache freshness and request pacing.
PRICE_CACHE_HOURS = 6
REQUEST_DELAY_SECONDS = 1.0

# AVAILABILITY FILTERS
MIN_INPUT_AVAILABILITY_PROB = 0.65
MIN_OUTPUT_SUCCESS_PROB = 0.55
MIN_OVERALL_TRADEUP_PROB = 0.50

# RISK CONTROL
MIN_RISK_ADJUSTED_SCORE = 0.02

# Scheduler mode: "once", "interval", or "times".
SCHEDULE_MODE = "once"
INTERVAL_MINUTES = 60
RUN_TIMES = ["02:00", "14:00"]
