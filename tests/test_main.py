"""Tests for churn_prediction.main."""

import matplotlib
matplotlib.use("Agg")

from unittest.mock import patch

from churn_prediction.main import main


def test_main_runs_end_to_end():
    """Smoke test: main() should complete without error."""
    with patch("matplotlib.pyplot.show"):  # suppress plot windows
        main()
