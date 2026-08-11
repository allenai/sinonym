import unittest

from sinonym.coretypes.routing_resolution import ResolutionAction, ResolutionReason
from sinonym.timo.interface import (
    Instance,
    Prediction,
    Predictor,
    PredictorConfig,
    ResolvedAuthorFields,
    SourceAuthorFields,
)


class TestIntegration(unittest.TestCase):
    """Integration contract for terminal, source-shaped author resolution."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor(config=PredictorConfig(), artifacts_dir=".")

    def test_structured_input_produces_directly_writable_fields(self):
        instance = Instance(
            pp_authors=[
                SourceAuthorFields(first_name="Steve", last_name="Blando", suffix="IV"),
                SourceAuthorFields(first_name="Li", last_name="Wei"),
            ],
            vys_other_names=["Jun Zhao", "Hui Li"],
        )

        (paper,) = self.predictor.predict_batch([instance])

        self.assertIsInstance(paper, Prediction)
        self.assertEqual(len(paper.authors), 2)
        self.assertTrue(all(isinstance(author, ResolvedAuthorFields) for author in paper.authors))
        self.assertEqual(paper.authors[0].suffix, "IV")
        self.assertEqual(paper.authors[1].last_name, "Li")
        self.assertIn("resolution_action", paper.authors[0].dict())
        self.assertNotIn("resolved_fields", paper.authors[0].dict())

    def test_empty_paper_still_emits_one_prediction(self):
        (paper,) = self.predictor.predict_batch([Instance(pp_authors=[])])
        self.assertEqual(paper.authors, [])

    def test_reviewed_non_person_is_machine_actionable(self):
        instance = Instance(
            pp_authors=[SourceAuthorFields(first_name="STADT", last_name="N\u00dcRNBERG")],
        )

        (paper,) = self.predictor.predict_batch([instance])

        resolved = paper.authors[0]
        self.assertIs(resolved.resolution_action, ResolutionAction.SUPPRESS)
        self.assertIs(resolved.resolution_reason, ResolutionReason.REVIEWED_NON_PERSON_PATTERN)

    def test_predict_batch_empty(self):
        self.assertEqual(self.predictor.predict_batch([]), [])
