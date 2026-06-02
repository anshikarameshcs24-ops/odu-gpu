import numpy as np
import torch

from src.data.tihm_binary_dataset import (
    TIHMBinaryPackedSequenceDataset,
    collate_tihm_binary_sequences,
)


def test_packed_dataset_returns_prebuilt_tensors(tmp_path):
    records = [
        {"trajectory": 0, "timestamps": ["t0", "t1"], "patient_id": "p1"},
        {"trajectory": 2, "timestamps": ["t0", "t1"], "patient_id": "p2"},
    ]
    pack_path = tmp_path / "train.npz"
    np.savez(
        pack_path,
        features=np.ones((2, 2, 3), dtype=np.float32),
        elevated=np.asarray([[0, 1], [1, 0]], dtype=np.float32),
        trajectory=np.asarray([0, 2], dtype=np.int64),
    )

    dataset = TIHMBinaryPackedSequenceDataset(records, pack_path)
    item = dataset[0]

    assert isinstance(dataset.features, torch.Tensor)
    assert isinstance(item["features"], torch.Tensor)
    assert item["features"].shape == (2, 3)
    assert item["length"].item() == 2


def test_tihm_binary_collate_keeps_tensor_batch_shapes(tmp_path):
    records = [
        {"trajectory": 0, "timestamps": ["t0", "t1"], "patient_id": "p1"},
        {"trajectory": 2, "timestamps": ["t0", "t1"], "patient_id": "p2"},
    ]
    pack_path = tmp_path / "train.npz"
    np.savez(
        pack_path,
        features=np.ones((2, 2, 3), dtype=np.float32),
        elevated=np.asarray([[0, 1], [1, 0]], dtype=np.float32),
        trajectory=np.asarray([0, 2], dtype=np.int64),
    )

    dataset = TIHMBinaryPackedSequenceDataset(records, pack_path)
    batch = collate_tihm_binary_sequences([dataset[0], dataset[1]])

    assert batch["features"].shape == (2, 2, 3)
    assert batch["elevated_labels"].shape == (2, 2)
    assert batch["trajectory"].tolist() == [0, 2]
    assert batch["lengths"].tolist() == [2, 2]
