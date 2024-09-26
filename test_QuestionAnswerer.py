import unittest
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from torch import tensor

from QuestionAnswerer import QuestionAnswerer

pad = 128002
class QuestionAnswererTests(unittest.TestCase):
    def setUp(self):
        self.qa = QuestionAnswerer('dummy', 'cpu', None)
        # self.qa.llm.tokenizer = MagicMock()
        self.qa.llm.tokenizer.pad_token_id = pad
        self.qa.llm.tokenizer.batch_decode = MagicMock(
            return_value = ['Hello how are you', 'Newline here', 'No stop string', '']
        )

    def test_winner(self):
        logits = tensor([
            [[0.0900, 0.2447, 0.6652], [0.6652, 0.2447, 0.0900], [0.2447, 0.6652, 0.0900]],
            [[0.2119, 0.2119, 0.5761], [0.2119, 0.2119, 0.5761], [0.2119, 0.2119, 0.5761]],
            [[0.6652, 0.2447, 0.0900], [0.2119, 0.5761, 0.2119], [0.5761, 0.2119, 0.2119]],
        ])

        expected_path = tensor([[2, 0, 1], [2, 2, 2], [0, 1, 0]])
        expected_probs = tensor([
            [0.6652, 0.6652, 0.6652],
            [0.5761, 0.5761, 0.5761],
            [0.6652, 0.5761, 0.5761],
        ])

        path, probs = self.qa.winner(logits)
        self.assertTrue(torch.equal(path, expected_path), msg = (path, expected_path))
        self.assertTrue(torch.allclose(probs, expected_probs), msg = (probs, expected_probs))

    def test_decode(self):
        path = tensor([
            [128000, 9906, 1268,  527,  499,    13,  358, 1097, 3815, 7060, 9901,  499,   13],
            [128000, 3648, 1074, 1618,  198, 54953,    0,   13, 1234, 1234, 1234, 1234, 1234],
            [128000, 2822, 3009,  925, 1234,  1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234],
            [128000,   13, 1234, 1234, 1234,  1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234],
        ])
        probs = tensor([
            [1., 3., 5., 7., 9., 11., 13., 15., 17., 19., 21., 23., 25.],
            [1., 3., 5., 7., 9., 11., 13., 15., 17., 19., 21., 23., 25.],
            [1., 3., 5., 7., 9., 11., 13., 15., 17., 19., 21., 23., 25.],
            [1., 3., 5., 7., 9., 11., 13., 15., 17., 19., 21., 23., 25.],
        ])

        expected_result = [
            'Hello how are you',
            'Newline here',
            'No stop string',
            '',
        ]
        expected_mean_probs = [5., 4., 13., 1.]

        result, mean_probs = self.qa.decode(path, probs)
        self.assertListEqual(expected_result, result)
        self.assertListEqual(expected_mean_probs, mean_probs)
