"""Exact coverage of ordinary split conformal intervals under covariate shift."""

from fractions import Fraction
from itertools import product
from math import ceil


def exact_coverage(calibration_probability, deployment_probability):
    """Enumerate nine binary calibration inputs and one future response."""
    calibration_size = 9
    target_coverage = Fraction(9, 10)
    rank = ceil((calibration_size + 1) * target_coverage)
    covered_probability = Fraction(0)
    total_probability = Fraction(0)

    for inputs in product((0, 1), repeat=calibration_size):
        # Y = X and the fixed predictor is zero, so residuals equal inputs.
        threshold = sorted(inputs)[rank - 1]
        number_of_ones = sum(inputs)
        calibration_weight = (
            calibration_probability ** number_of_ones
            * (1 - calibration_probability)
            ** (calibration_size - number_of_ones)
        )

        for future_response in (0, 1):
            future_weight = (
                deployment_probability
                if future_response == 1
                else 1 - deployment_probability
            )
            outcome_weight = calibration_weight * future_weight
            total_probability += outcome_weight
            if -threshold <= future_response <= threshold:
                covered_probability += outcome_weight

    assert total_probability == 1
    return covered_probability


def main():
    calibration_probability = Fraction(1, 10)
    print("Calibration size = 9; target coverage = 90%")
    print("Deployment q | Exact coverage | Percentage")

    for deployment_probability in (
        Fraction(1, 10), Fraction(3, 10),
        Fraction(1, 2), Fraction(4, 5),
    ):
        enumerated = exact_coverage(
            calibration_probability, deployment_probability
        )
        derived = (
            1 - deployment_probability * (1 - calibration_probability) ** 9
        )
        assert enumerated == derived
        print(
            f"{float(deployment_probability):.1f}"
            f" | {float(enumerated):.10f}"
            f" | {float(enumerated):.2%}"
        )

    # Check validity for several same-population cases, including the boundaries.
    for numerator in range(11):
        probability = Fraction(numerator, 10)
        assert exact_coverage(probability, probability) >= Fraction(9, 10)

    print("Exact enumeration, total probability, and same-population checks passed.")


if __name__ == "__main__":
    main()
