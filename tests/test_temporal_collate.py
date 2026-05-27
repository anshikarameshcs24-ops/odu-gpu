import numpy as np
import torch

from src.data.temporal_dataset import collate_sequences


def test_collate_sequences_pads_to_max_length():
    batch = [
        {
            "embeddings": torch.tensor(np.ones((2, 512)), dtype=torch.float32),
            "cmai_labels": torch.zeros(2, 29),
            "risk_levels": torch.tensor([0, 1]),
            "trajectory": torch.tensor(1),
            "length": torch.tensor(2),
        },
        {
            "embeddings": torch.tensor(np.ones((3, 512)), dtype=torch.float32),
            "cmai_labels": torch.zeros(3, 29),
            "risk_levels": torch.tensor([0, 1, 2]),
            "trajectory": torch.tensor(2),
            "length": torch.tensor(3),
        },
    ]

    output = collate_sequences(batch)
    assert output["embeddings"].shape == (2, 3, 512)
    assert output["cmai_labels"].shape == (2, 3, 29)
    assert output["risk_levels"].shape == (2, 3)
    assert output["lengths"].tolist() == [2, 3]

