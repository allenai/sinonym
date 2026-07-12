import unittest

from sinonym.timo.interface import (
    Instance,
    Prediction,
    PredictionV2,
    Predictor,
    PredictorConfig,
    PredictorV2,
    RoutedPaperPrediction,
    RoutedPaperPredictionV2,
    RoutedPrediction,
    RoutingInstance,
    RoutingPredictor,
    RoutingPredictorV2,
)


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
        assert results[0].format_pattern is not None
        assert results[1].format_pattern is not None
        self.assertEqual(
            results[0].format_pattern.dominant_format,
            results[1].format_pattern.dominant_format,
        )

    def test_predict_batch_empty(self):
        self.assertEqual(self.predictor.predict_batch([]), [])

    def test_score_name_batch(self):
        names = ["Li Wei", "Wang Weiming"]
        summary = self.predictor.score_name_batch(names)
        self.assertEqual(summary.names, names)
        self.assertEqual(len(summary.results), 2)
        self.assertEqual(len(summary.confidences), 2)
        self.assertIsNotNone(summary.format_pattern.dominant_format)

    def test_score_name_batch_tuned_threshold(self):
        names = ["Li Wei", "Wang Weiming"]
        summary = self.predictor.score_name_batch(names, format_threshold=0.9)
        self.assertEqual(len(summary.results), 2)
        if summary.format_pattern.threshold_met:
            self.assertGreaterEqual(summary.format_pattern.decision_confidence, 0.9)

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


class TestRoutingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = RoutingPredictor(config=PredictorConfig(), artifacts_dir=".")

    def test_route_with_vys_pool(self):
        # paper authors must be the leading slice of the pool
        pp = ["Yue Lin", "Wei Wang"]
        pool = pp + ["Jun Zhao", "Hui Li"]
        results = self.predictor.predict_batch([RoutingInstance(pp_names=pp, vys_pool_names=pool)])
        self.assertEqual(len(results), 1)  # one prediction per instance (timo 1:1 contract)
        paper = results[0]
        self.assertIsInstance(paper, RoutedPaperPrediction)
        self.assertEqual(len(paper.authors), 2)  # one RoutedPrediction per pp author
        for r in paper.authors:
            self.assertIsInstance(r, RoutedPrediction)
            self.assertIn(r.router_prediction, {"pp", "vys", "abstain", "not_person"})
            self.assertIsNotNone(r.vys)  # vys candidate present when a pool is given

    def test_route_pp_only_fallback(self):
        results = self.predictor.predict_batch([RoutingInstance(pp_names=["Li Wei"])])
        self.assertEqual(len(results), 1)
        authors = results[0].authors
        self.assertEqual(len(authors), 1)
        self.assertIsNone(authors[0].vys)  # PP-only fallback: no venue pool
        self.assertIsNone(authors[0].input_order_candidate)

    def test_predict_batch_is_one_to_one(self):
        instances = [
            RoutingInstance(pp_names=["Li Wei", "Wang Weiming"]),
            RoutingInstance(pp_names=["Zhang San"]),
        ]
        results = self.predictor.predict_batch(instances)
        self.assertEqual(len(results), len(instances))  # 1:1, paper boundaries preserved
        self.assertEqual([len(p.authors) for p in results], [2, 1])

    def test_empty_paper_still_emits_one_prediction(self):
        # an instance with no authors must not vanish from the output stream
        results = self.predictor.predict_batch([RoutingInstance(pp_names=[])])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].authors, [])

    def test_predict_batch_empty(self):
        self.assertEqual(self.predictor.predict_batch([]), [])


class TestIntegrationV2(TestIntegration):
    """Integration contract for the flat TIMO v2 variant."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = PredictorV2(config=PredictorConfig(), artifacts_dir=".")

    def test_non_chinese_canonical_name(self):
        (result,) = self.predictor.predict_batch([Instance(name="Dr. Steve Marsh PhD")])

        self.assertIsInstance(result, PredictionV2)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.canonical_name)
        assert result.canonical_name is not None
        self.assertEqual(result.canonical_name.text, "Steve Marsh")


class TestRoutingIntegrationV2(TestRoutingIntegration):
    """Integration contract for the routed TIMO v2 variant."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = RoutingPredictorV2(config=PredictorConfig(), artifacts_dir=".")

    def test_non_chinese_canonical_name(self):
        (paper,) = self.predictor.predict_batch([RoutingInstance(pp_names=["Steve Blando IV"])])

        self.assertIsInstance(paper, RoutedPaperPredictionV2)
        canonical_name = paper.authors[0].canonical_name
        self.assertIsNotNone(canonical_name)
        assert canonical_name is not None
        self.assertEqual(canonical_name.text, "Steve Blando IV")
        self.assertEqual(canonical_name.normalized.suffix, "IV")
