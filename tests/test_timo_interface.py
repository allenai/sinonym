import unittest

from sinonym.timo.interface import Instance, Prediction, Predictor, PredictorConfig


class TestTimoInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor(config=PredictorConfig(), artifacts_dir=".")

    def test_single_chinese_name(self):
        results = self.predictor.predict_batch([Instance(name="Li Wei")])
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertTrue(r.success)
        self.assertIsNotNone(r.surname)
        self.assertIsNotNone(r.given_name)
        self.assertIsNotNone(r.confidence)
        self.assertIsNone(r.error_message)

    def test_chinese_name_characters(self):
        results = self.predictor.predict_batch([Instance(name="巩俐")])
        self.assertTrue(results[0].success)
        self.assertIsNotNone(results[0].surname)

    def test_chinese_name_comma_format(self):
        results = self.predictor.predict_batch([Instance(name="Zhang, Ming")])
        r = results[0]
        self.assertTrue(r.success)
        self.assertIsNotNone(r.surname)
        self.assertIsNotNone(r.given_name)

    def test_non_chinese_name(self):
        results = self.predictor.predict_batch([Instance(name="John Smith")])
        r = results[0]
        self.assertFalse(r.success)
        self.assertIsNotNone(r.error_message)
        self.assertIsNone(r.given_name)
        self.assertIsNone(r.surname)

    def test_empty_string(self):
        results = self.predictor.predict_batch([Instance(name="")])
        self.assertFalse(results[0].success)

    def test_whitespace_only(self):
        results = self.predictor.predict_batch([Instance(name="   ")])
        self.assertFalse(results[0].success)

    def test_batch_with_context(self):
        instances = [
            Instance(name="Li Wei"),
            Instance(name="Wang Weiming"),
            Instance(name="John Smith"),
        ]
        results = self.predictor.predict_batch(instances)
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0].success)
        self.assertTrue(results[1].success)
        self.assertFalse(results[2].success)

    def test_prediction_is_pydantic_model(self):
        results = self.predictor.predict_batch([Instance(name="Li Wei")])
        r = results[0]
        self.assertIsInstance(r, Prediction)
        as_dict = r.dict()
        self.assertIn("success", as_dict)
        self.assertIn("error_message", as_dict)
        self.assertIn("given_name", as_dict)
        self.assertIn("surname", as_dict)
        self.assertIn("middle_name", as_dict)
        self.assertIn("confidence", as_dict)
        self.assertIn("format_pattern", as_dict)

    def test_prediction_json_serialization(self):
        results = self.predictor.predict_batch([Instance(name="Li Wei")])
        json_str = results[0].json()
        self.assertIn('"success": true', json_str)
        self.assertIn('"confidence":', json_str)

    def test_format_pattern_exposes_batch_decision_confidence(self):
        summary = self.predictor.score_name_batch(["J. Liu", "Jing Wan"])
        pattern = summary.format_pattern

        self.assertEqual(pattern.confidence, 0.5)
        self.assertGreaterEqual(pattern.decision_confidence, pattern.confidence)
        self.assertEqual(pattern.voting_count, pattern.surname_first_count + pattern.given_first_count)
        self.assertTrue(pattern.threshold_met)

    def test_compound_surname(self):
        results = self.predictor.predict_batch([Instance(name="Ouyang Ming")])
        if results[0].success:
            self.assertIsNotNone(results[0].surname)

    def test_empty_batch(self):
        results = self.predictor.predict_batch([])
        self.assertEqual(len(results), 0)
