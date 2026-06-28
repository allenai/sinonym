import unittest

from sinonym.timo.interface import Instance, Prediction, Predictor, PredictorConfig


class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor(config=PredictorConfig(), artifacts_dir=".")

    def test_chinese_name(self):
        results = self.predictor.predict_batch([Instance(name="Li Wei")])
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Prediction)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].given_name, "Wei")
        self.assertEqual(results[0].surname, "Li")

    def test_non_chinese_name(self):
        results = self.predictor.predict_batch([Instance(name="John Smith")])
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIsNotNone(results[0].error_message)

    def test_batch_superset_output(self):
        instances = [Instance(name="Li Wei"), Instance(name="Wang Weiming")]
        results = self.predictor.predict_batch(instances)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, Prediction)
            self.assertTrue(r.success)
            # superset: every Prediction carries confidence + format_pattern
            self.assertIsNotNone(r.confidence)
            self.assertIsNotNone(r.format_pattern)
        # shared batch pattern replicated onto each row
        self.assertEqual(
            results[0].format_pattern.dominant_format,
            results[1].format_pattern.dominant_format,
        )

    def test_predict_batch_empty(self):
        self.assertEqual(self.predictor.predict_batch([]), [])

    def test_score_name_batch(self):
        names = ["Li Wei", "Wang Weiming"]
        summary = self.predictor.score_name_batch(names)
        self.assertEqual(summary.names, tuple(names))
        self.assertEqual(len(summary.results), 2)
        self.assertEqual(len(summary.confidences), 2)
        self.assertIsNotNone(summary.format_pattern.dominant_format)

    def test_score_name_batch_tuned_threshold(self):
        names = ["Li Wei", "Wang Weiming"]
        summary = self.predictor.score_name_batch(names, format_threshold=0.9)
        self.assertEqual(len(summary.results), 2)
        self.assertEqual(
            summary.format_pattern.threshold_met,
            summary.format_pattern.decision_confidence >= 0.9,
        )

    def test_detect_batch_format(self):
        pattern = self.predictor.detect_batch_format(["Zhang Wei", "Li Ming", "Wang Xiaoli"])
        self.assertIn(pattern.dominant_format, {"surname_first", "given_first", "mixed"})

    def test_analyze_name_batch_full(self):
        result = self.predictor.analyze_name_batch(["Li Wei", "Wang Weiming"])
        self.assertEqual(len(result.results), 2)
        self.assertEqual(len(result.individual_analyses), 2)
        self.assertEqual(len(result.name_order_evidence), 2)
        self.assertEqual(result.name_order_evidence[0].raw_name, "Li Wei")
        self.assertEqual(result.name_order_evidence[0].selected_format, "surname_first")
