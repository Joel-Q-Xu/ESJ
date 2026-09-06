"""OpenFHE strict interval filter: 5 < part.SIZE < 20."""

from openfhe_suite import run_single_filter


if __name__ == "__main__":
    run_single_filter("cmp", lower_bound=5)
