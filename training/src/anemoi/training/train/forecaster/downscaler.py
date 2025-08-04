# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Mapping
from operator import itemgetter
from hydra.utils import instantiate
import torch
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.forecaster import GraphForecaster

LOGGER = logging.getLogger(__name__)


class GraphDownscaler(GraphForecaster):
    """Graph neural network downscaler for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        truncation_data: dict,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network interpolator.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__(
            config=config,
            graph_data=graph_data,
            truncation_data=truncation_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )

        reader_group_size = self.config.dataloader.read_group_size
        self.grid_indices = instantiate(
            self.config.model_dump(by_alias=True).dataloader.grid_indices_target,
            reader_group_size=reader_group_size,
        )
        self.grid_indices.setup(graph_data)

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        del batch_idx

        metrics = {}
        y_preds = []

        x, y = batch
        x = self.model.pre_processors(x)
        y = self.model.pre_processors(y)

        # Delayed scalers need to be initialized after the pre-processors once
        if self.is_first_step:
            self.define_delayed_scalers()
            self.is_first_step = False

        x = x[:, :, ..., self.data_indices.data.input.full]
        y = y[:, :, ..., self.data_indices.data.output.full]

        y_pred = self(x)

        loss, metrics_next = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                0,
                training_mode=True,
                validation_mode=validation_mode,
                use_reentrant=False,
        )

        metrics.update(metrics_next)
        y_preds.append(y_pred)

        return loss, metrics, y_preds

    def allgather_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Allgather the batch-shards across the reader group.

        Parameters
        ----------
        batch : torch.Tensor
            Batch-shard of current reader rank

        Returns
        -------
        torch.Tensor
            Allgathered (full) batch
        """
        grid_shard_shapes = self.grid_indices.shard_shapes
        grid_size = self.grid_indices.grid_size

        if grid_size == batch[0].shape[self.grid_dim] or self.reader_group_size == 1:
            return batch  # already have the full grid

        else:
            raise NotImplementedError("Sharding strategy for downscaling not implemented yet")

        return batch


