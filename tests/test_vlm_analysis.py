import os
import shutil
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from pathlib import Path
import math
from PIL import Image
import sys
import os
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from vlm_analysis import ResultsManager, ImageAnalyzer


class TestResultsManager(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        np.random.seed(42)

        # Setup test directory structure
        self.seq_id = "test_seq"
        self.base_path = Path("images/masks", self.seq_id)
        self.image_ids = ["test_img_0001", "test_img_0002"]
        self.categories = ["building", "vegetation"]

        # Create folders for each category and image
        for img_id in self.image_ids:
            for category in self.categories:
                img_dir = self.base_path / img_id / category
                img_dir.mkdir(parents=True, exist_ok=True)

                # Save a random 2048x1148 RGB image
                array = (np.random.rand(1148, 2048, 3) * 255).astype(np.uint8)
                img = Image.fromarray(array)
                img.save(img_dir / f"{img_id}.jpg")

        # Create minimal metadata.csv
        self.metadata = pd.DataFrame({
            "image_id": [self.image_ids[0]] * 2 + [self.image_ids[1]] * 2,  # Each image has two rows (for two combinations)
            "label": ["building", "vegetation"] * len(self.image_ids),
            "building": [1, 0, 1, 0],
            "vegetation": [0, 1, 0, 1],
            "building_p": [0.3, 0.0, 0.25, 0.0],
            "vegetation_p": [0.0, 0.4, 0.0, 0.3]
        })
        metadata_path = self.base_path / "metadata.csv"
        self.metadata.to_csv(metadata_path, index=False)
        
        self.results_manager = ResultsManager(self.base_path)

    def tearDown(self):
        # Clean up the entire test sequence folder
        shutil.rmtree(Path("images/masks") / self.seq_id)
        
    def test_add_unmasked_returns_df(self):
        # setup input dataframe
        results = self.metadata.copy()
        results['score'] = 70
        results['response'] = 'a'
        # setup data to add
        scores = {'unmasked':70}
        responses = {'unmasked':'a'}
        image_id = 'test_123'

        output = self.results_manager.add_unmasked_row(results, image_id, scores, responses)
        msg = "add_unmasked does not return a dataframe."
        self.assertIsInstance(output, pd.DataFrame, msg=msg)

    def test_add_unmasked_adds_one_row(self):
        # setup input dataframe
        results = self.metadata.copy()
        results['score'] = 70
        results['respnse'] = 'a'
        # setup data to add
        scores = {'unmasked':70}
        responses = {'unmasked':'a'}
        image_id = 'test_123'
        # get output
        output = self.results_manager.add_unmasked_row(results, image_id, scores, responses)
        msg = "add_unmasked does not add single row."
        self.assertEqual(len(output), len(results) + 1, msg=msg)
        
    def test_add_image_results_updates_results(self):
        
        image_id = self.image_ids[0]
        scores = {'building':70, 'vegetation':70, 'unmasked':80}
        responses = {'building':'a', 'vegetation':'b', 'unmasked':'c'}
        initial_length = len(self.results_manager.results)
        
        self.results_manager.add_image_results(image_id, scores, responses)
        
        msg = "add_image_results does not add to results"
        self.assertEqual(len(self.results_manager.results), initial_length + 1, msg=msg)
        
    def test_add_image_results_updates_df_correctly(self):
        image_id = self.image_ids[0]
        scores = {'building': 70, 'vegetation': 70, 'unmasked': 80}
        responses = {'building': 'a', 'vegetation': 'b', 'unmasked': 'c'}

        self.results_manager.add_image_results(image_id, scores, responses)
        out_df = self.results_manager.results[-1]

        # Check that the correct number of rows were added
        self.assertEqual(len(out_df), len(scores))

        # Check that all expected rows exist with correct values
        for label, expected_score in scores.items():
            row = out_df[(out_df.image_id == image_id) & (out_df.label == label)]
            self.assertEqual(len(row), 1, f"Expected one row for label {label}")
            self.assertEqual(row.iloc[0]['score'], expected_score)
            self.assertEqual(row.iloc[0]['response'], responses[label])
            
            
class TestImageAnalyzer(unittest.TestCase):
    
    def setUp(self):
        # Set random seed for reproducibility
        np.random.seed(42)

        # Setup test directory structure
        self.seq_id = "test_seq"
        self.base_path = Path("images/masks", self.seq_id)
        self.image_ids = ["test_img_0001", "test_img_0002"]
        self.categories = ["building", "vegetation"]

        # Create folders for each category and image
        for img_id in self.image_ids:
            for category in self.categories:
                img_dir = self.base_path / img_id
                img_dir.mkdir(parents=True, exist_ok=True)

                # Save a random 2048x1148 RGB image
                array = (np.random.rand(1148, 2048, 3) * 255).astype(np.uint8)
                img = Image.fromarray(array)
                img.save(img_dir / f"{category}.jpg")

        # Create minimal metadata.csv
        self.metadata = pd.DataFrame({
            "image_id": [self.image_ids[0]] * 2 + [self.image_ids[1]] * 2,  # Each image has two rows (for two combinations)
            "label": ["building", "vegetation"] * len(self.image_ids),
            "building": [1, 0, 1, 0],
            "vegetation": [0, 1, 0, 1],
            "building_p": [0.3, 0.0, 0.25, 0.0],
            "vegetation_p": [0.0, 0.4, 0.0, 0.3]
        })
        metadata_path = self.base_path / "metadata.csv"
        self.metadata.to_csv(metadata_path, index=False)
        
        self.model = MockInternVL3Model()
        self.tokenizer = MockTokenizer()

    def tearDown(self):
        # Clean up the entire test sequence folder
        shutil.rmtree(Path("images/masks") / self.seq_id)   
        
    def test_analyze_image_calls_model_and_returns_response(self):
        analyzer = ImageAnalyzer(model=self.model, tokenizer=self.tokenizer, model_config={})
        dummy_path = Path("fake_path.jpg")
        dummy_question = "Testing."
        dummy_tensor = torch.randn(3, 448, 448)

        with patch("vlm_analysis.load_image", return_value=dummy_tensor) as mock_loader:

            response = analyzer.analyze_image(dummy_path, dummy_question)
            expected = 'This is a test response for a mocked model. \n ```[{"estimated_score": 70}]```'
            self.assertEqual(expected, response)
            
    def test_analyze_all_masks_return_expected(self):
        analyzer = ImageAnalyzer(model=self.model, tokenizer=self.tokenizer, model_config={})
        
        image_dir = self.base_path / self.image_ids[0]
        question = 'dummy question'
        
        scores_out, responses_out = analyzer.analyze_all_masks(image_dir, question)
        
        expected_scores = {"building":70, "vegetation":70}
        expected_responses = {"building":'This is a test response for a mocked model. \n ```[{"estimated_score": 70}]```', 
                              "vegetation":'This is a test response for a mocked model. \n ```[{"estimated_score": 70}]```'}
        self.assertDictEqual(scores_out, expected_scores)
        self.assertDictEqual(responses_out, expected_responses)
        
    
        
            

class MockInternVL3Model:
    def chat(self, tokenizer, pixel_values, question, model_config, history=None, return_history=True):
        # Fake reasoning and JSON output
        mock_score = 70
        mock_reason = "This is a test response for a mocked model."
        mock_response = mock_reason + ' \n ```[{"estimated_score": ' + str(mock_score) + '}]```'
        return mock_response


class MockTokenizer: 
    def __init__(self):
        self.eos_token_id = 0  # Fake end-of-sequence token ID


if __name__ == '__main__':
    unittest.main()