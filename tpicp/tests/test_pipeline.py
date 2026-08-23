import unittest

from tpicp.pipeline import PipelineError, select_best_detection


class PipelineHelpersTest(unittest.TestCase):
    def test_selects_highest_scoring_detection(self):
        detections = [
            {"category_id": 1, "score": 0.2, "R": [1], "t": [2]},
            {"category_id": 1, "score": 0.8, "R": [3], "t": [4]},
        ]
        selected = select_best_detection(detections)
        self.assertEqual(selected["score"], 0.8)

    def test_filters_requested_category(self):
        detections = [
            {"category_id": 1, "score": 0.9, "R": [1], "t": [2]},
            {"category_id": 5, "score": 0.6, "R": [3], "t": [4]},
        ]
        selected = select_best_detection(detections, category_id=5)
        self.assertEqual(selected["category_id"], 5)

    def test_rejects_empty_detections(self):
        with self.assertRaises(PipelineError):
            select_best_detection([])


if __name__ == "__main__":
    unittest.main()
