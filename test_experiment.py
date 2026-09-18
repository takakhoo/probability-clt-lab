import unittest
import numpy as np
from scipy.stats import binom
from experiment import intervals, exact_coverage, estimator_risks


class ProbabilityTests(unittest.TestCase):
    def test_wilson_boundary_and_symmetry(self):
        low, high = intervals(np.arange(11), 10)
        self.assertEqual(low[0], 0)
        self.assertEqual(high[-1], 1)
        self.assertGreater(high[0], 0.27)
        np.testing.assert_allclose(low, 1 - high[::-1], atol=1e-15)
        self.assertEqual(exact_coverage(10, 0, "wilson"), 1)
        self.assertEqual(exact_coverage(10, 1, "wilson"), 1)

    def test_exact_risk_matches_enumeration(self):
        for p in [0, 0.01, 0.5, 0.9, 1]:
            k = np.arange(21)
            probabilities = binom.pmf(k, 20, p)
            expected = [np.dot(probabilities, (k / 20 - p) ** 2),
                        np.dot(probabilities, ((k + 6) / 36 - p) ** 2)]
            np.testing.assert_allclose(estimator_risks(20, p), expected, atol=1e-15)

    def test_boundary_failure_of_wald(self):
        self.assertLess(exact_coverage(50, 0.01, "wald"), 0.4)
        self.assertGreater(exact_coverage(50, 0.01, "wilson"), 0.9)

    def test_prior_bias_is_not_always_an_improvement(self):
        mle, bayes = estimator_risks(10, 0.9)
        self.assertGreater(bayes, mle)
        mle, bayes = estimator_risks(10, 0.375)
        self.assertLess(bayes, mle)

    def test_input_validation(self):
        for count, n in [(0, 0), (11, 10), (0.5, 10), (float("nan"), 10)]:
            with self.assertRaises(ValueError):
                intervals(count, n)
        with self.assertRaises(ValueError):
            intervals(1, 10, "unknown")
        with self.assertRaises(ValueError):
            exact_coverage(10, 2, "wilson")


if __name__ == "__main__":
    unittest.main()
