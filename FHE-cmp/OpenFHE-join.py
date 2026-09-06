"""OpenFHE only-join, join+eq, and join+(5 < SIZE < 20)."""

from openfhe_suite import run_join_suite


if __name__ == "__main__":
    run_join_suite(lower_bound=5)
