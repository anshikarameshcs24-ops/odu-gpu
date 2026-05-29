"""Binary elevated-risk TIHM temporal model."""

from __future__ import annotations

from src.models.tihm_sensor_model import TIHMSensorTemporalModel


class TIHMBinaryTemporalModel(TIHMSensorTemporalModel):
    def __init__(
        self,
        feature_dim: int,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            num_risk=1,
            num_layers=num_layers,
            dropout=dropout,
        )
