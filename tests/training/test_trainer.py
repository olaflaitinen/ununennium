"""Tests for trainer."""

import pytest
import torch
import torch.nn as nn

from ununennium.training import Trainer


class TestTrainer:
    def test_trainer_init(self):
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.MSELoss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=None,
        )
        assert trainer.model is model
