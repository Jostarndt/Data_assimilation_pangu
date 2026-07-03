import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tensordict.tensordict import TensorDict
from torchmetrics import Metric

from datetime import datetime
from typing import Any, Callable, Dict, List



def compute_lat_weights(latitude_resolution: int) -> torch.tensor:
    """Compute latitude coefficients for latititude weighted metrics.

    Assumes latitude coordinates are equidistant and ordered from -90 to 90.

    Args:
        latitude_resolution: latititude dimension size.
    """
    if latitude_resolution == 1:
        return torch.tensor(1.0)
    lat_coeffs_equi = torch.tensor(
        [
            torch.cos(x)
            for x in torch.arange(
                -torch.pi / 2, torch.pi / 2 + 1e-6, torch.pi / (latitude_resolution - 1)
            )
        ]
    )
    lat_coeffs_equi = lat_coeffs_equi / lat_coeffs_equi.mean()
    return lat_coeffs_equi[:, None]


def compute_lat_weights_weatherbench(latitude_resolution: int) -> torch.tensor:
    """Calculate the area overlap as a function of latitude.
    The weatherbench version gives slightly different coeffs.
    """
    latitudes = torch.linspace(-90, 90, latitude_resolution)
    points = torch.deg2rad(latitudes)
    pi_over_2 = torch.tensor([torch.pi / 2], dtype=torch.float32)
    bounds = torch.concatenate([-pi_over_2, (points[:-1] + points[1:]) / 2, pi_over_2])
    upper = bounds[1:]
    lower = bounds[:-1]
    # normalized cell area: integral from lower to upper of cos(latitude)
    weights = torch.sin(upper) - torch.sin(lower)
    weights = weights / weights.mean()
    return weights[:, None]




class TensorDictMetricBase(Metric):
    """Wrapper around metric to enable handling of targets and preds that are TensorDicts.

    Assumes metric should accept tensor target and pred.
    Keeps track of a metric instantiation per item in the TensorDict.

    Warning: not compatible with metric.forward() - only use update() and compute().
    See https://github.com/Lightning-AI/torchmetrics/issues/987#issuecomment-2419846736.
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: mapping of key to metric.
                Key should match the key in the TensorDict.
                Metric should be an instantiation of a metric class that accepts tensors.

        Example:
            preds = TensorDict(level=torch.tensor(...), surface=torch.tensor(...))
            targets = TensorDict(level=torch.tensor(...), surface=torch.tensor(...))
            metric = TensorDictMetricBase(level=BrierSkillScore(), surface=BrierSkillScore())
            metric.update(targets, preds)
        """
        super().__init__()
        self.metrics = nn.ModuleDict(kwargs)

    def update(self, targets: TensorDict, preds: TensorDict | List[TensorDict], timestamp: torch.Tensor | None = None) -> None:

        """Update internal metrics.

        Returns:
            None
        """
        if isinstance(preds, list):
            preds = torch.stack(preds, dim=1)

        for key, metric in self.metrics.items():
            if key == 'surface_seeps':
                metric.update(targets=targets['surface'], preds=preds['surface'], timestamp=timestamp)
            else:
                metric.update(targets=targets[key], preds=preds[key], timestamp=timestamp)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Return aggregated collections of the computed metrics.

        Elements from each metric are aggregated. Handles multiple return values per metric.
        Assumes all metrics return the same number of outputs.
        """
        aggregated_outputs = []

        for key, metric in self.metrics.items():
            # Collect returned values from each metric.
            outputs = metric.compute()
            if not isinstance(outputs, tuple):
                outputs = [outputs]
            for i, output in enumerate(outputs):
                if isinstance(output, dict):
                    if len(aggregated_outputs) - 1 < i:
                        aggregated_outputs.append({})
                    if aggregated_outputs[i].keys().isdisjoint(output.keys()):
                        aggregated_outputs[i].update(output)
                    else:
                        aggregated_outputs[i].update({f"{k}_{key}": v for k, v in output.items()})

        if len(aggregated_outputs) == 1:
            return aggregated_outputs[0]
        return aggregated_outputs

    def reset(self):
        """
        Reset states of all metrics.
        """
        for metric in self.metrics.values():
            metric.reset()



class LabelDictWrapper(Metric):
    """Wrapper class around metric for extracting metric outputs into a labelled dictionary.
    Helpful for WandB which needs to log single values.

    Expects the metric to return a dictionary holding computed metrics:
        - keys: metric_name
        - values: torch tensors with shape (..., *(variable_index))
                  variable_index is passed in with param `variable_indices`

    LabelDictWrapper returns a dictionary of computed metrics:
        - keys: <metric_name>_<variable_name>
        - value: torch tensors with shape (...)

    Warning: this class is not compatible with forward(), only use update() and compute().
    See https://github.com/Lightning-AI/torchmetrics/issues/987#issuecomment-2419846736.

    Example:
        metric = LabelDictWrapper(EnsembleMetrics(preprocess=preprocess_fn),
                              variable_indices=dict(T2m_24h=(0, 2, 0), T2m_48h=(1, 2, 0), U10_24h=(0, 0, 0)), U10_48h=(1, 0, 0)))
        targets, preds = torch.tensor(batch, timedelta, var, lev, lat, lon), torch.tensor(batch, nmem, timedelta, var, lev, lat, lon)
        metric.update(targets, preds)
        labeled_dict = metric.compute()  # EnsembleMetrics returns {"mse": torch.tensor(timedelta, var, lev) }
        labelled_dict = {"mse_T2m_24h": ..., "mse_T2m_48h": ..., "mse_U10_24h": ..., "mse_U10_48h": ...}

    Args:
        metric: base metric that should be wrapped. It is assumed that the metric outputs a
            dict mapping metric name to tensors that have shape (..., *(variable_index)).
        variable_indices: Mapping from variable name to index (ie. var, lev) into tensor holding computed metric.
                ie. dict(T2m=(2, 0), U10=(0, 0), V10=(1, 0), SP=(3, 0)).
    """

    def __init__(
        self,
        metric: Metric,
        variable_indices: Dict[str, tuple],
    ):
        super().__init__()
        if not isinstance(metric, Metric):
            raise ValueError(
                f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {metric}"
            )
        self.metric = metric
        self.variable_indices = variable_indices

    def _convert(self, raw_metric_dict: Dict[str, Tensor]):
        # Label metrics.
        labeled_dict = dict()
        for var, index in self.variable_indices.items():
            for metric_name, metric in raw_metric_dict.items():
                labeled_dict[f"{metric_name}_{var}"] = metric.__getitem__((..., *index))
        return labeled_dict

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.metric.update(*args, **kwargs)

    def compute(self) -> Dict[str, Tensor]:
        return self._convert(self.metric.compute())

    def reset(self) -> None:
        """Reset metric."""
        self.metric.reset()
        super().reset()


def add_timedelta_index(
    variable_indices: dict[str, tuple],
    lead_time_hours: None | int = None,
    rollout_iterations: None | int = None,
):
    """Add prediction_timedelta dimension to variable indices for LabelDictWrapper.

    For example: if variable indexes are (var, lev).
    Returns indexes with (timedelta, var, lev).
    Means that LabelDictWrapper expects metric to return metrics with shape (..., timedelta, var, lev).

    Args:
        variable_indices: Mapping from variable name to index (ie. var, lev).
        lead_time_hours: time delta between timesteps in multistep rollout.
        rollout_iterations: Number of rollout iterations in multistep predictions. ie. Size of prediction_timdelta dimension.
    """
    if lead_time_hours is None or rollout_iterations is None:
        return variable_indices
    indices = {}
    for var, index in variable_indices.items():
        for i in range(rollout_iterations):
            lead_time = lead_time_hours * (i + 1)
            indices[f"{var}_{lead_time}h"] = (i, *index)
    return indices


class MetricBase:
    """Implement latitude-weighted base functions."""

    def __init__(
        self,
        compute_lat_weights_fn: Callable[[int], torch.tensor] = compute_lat_weights_weatherbench,
    ):
        """
        Args:
            variable_indices: dict used to extract indices from output tensor.
            compute_lat_weights_fn: Function to compute latitude weights given latitude shape.
                Used for error and variance calculations. Expected shape of weights: [..., lat, 1].
        """
        super().__init__()
        self.compute_lat_weights_fn = compute_lat_weights_fn

    def wmse(self, x: torch.Tensor, y: torch.Tensor | int = 0):
        """Latitude weighted mse error.

        Args:
            x: preds with shape (..., lat, lon)
            y: targets with shape (..., lat, lon)
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return (x - y).pow(2).mul(lat_coeffs).mean((-2, -1))

    def wmae(self, x: torch.Tensor, y: torch.Tensor | int = 0):
        """Latitude weighted mae error.

        Args:
            x: preds with shape (..., lat, lon)
            y: targets with shape (..., lat, lon)
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return (x - y).abs().mul(lat_coeffs).mean((-2, -1))

    def wvar(self, x: torch.Tensor, dim: int = 1):
        """Latitude weighted variance along axis.

        Args:
            x: preds with shape (..., lat, lon)
            dim: over which dimension to compute variance.
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return x.var(dim).mul(lat_coeffs).mean((-2, -1))

    def weighted_mean(self, x: torch.Tensor):
        """Latitude weighted mean over grid.

        Args:
            x: preds with shape (..., lat, lon)
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return x.mul(lat_coeffs).mean((-2, -1))



class DeterministicRMSE(Metric, MetricBase):
    """
    Metrics for deterministic prediction

    """

    def __init__(
        self,
        data_shape: tuple,
        compute_lat_weights_fn: Callable[[int], torch.tensor] = compute_lat_weights_weatherbench,
    ):
        """
        Args:

            variable_indices: Mapping from variable name to (var, lev) index into tensor holding computed metric.
                ie. dict(T2m=(2, 0), U10=(0, 0), V10=(1, 0), SP=(3, 0)).
                Will index into metric tensor with (..., *index) to handle extra dimensions such as multistep.
            compute_lat_weights_fn: Function to compute latitude weights given latitude shape.
                Used for error and variance calculations. Expected shape of weights: [..., lat, 1].
                See function example in metric_base.MetricBase.
                Default function assumes latitudes are ordered -90 to 90.
        """
        Metric.__init__(self)
        MetricBase.__init__(
            self,
            compute_lat_weights_fn=compute_lat_weights_fn,
            # variable_indices=variable_indices,
            # lead_time_hours=lead_time_hours,
            # rollout_iterations=rollout_iterations,
        )

        # Call `self.add_state`for every internal state that is needed for the metrics computations.
        # `dist_reduce_fx` indicates the function that should be used to reduce.
        self.add_state("nsamples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("mse", default=torch.zeros(data_shape), dist_reduce_fx="sum")
        self.add_state(
            "rmse_before_time_avg", default=torch.zeros(data_shape), dist_reduce_fx="sum"
        )

    def update(self, targets: torch.Tensor, preds: torch.Tensor, timestamp: torch.Tensor | None=None) -> None:
        """Update internal state with a batch of targets and predictions.

        Expects inputs to this function to be denormalized.

        Args:
            targets: Target tensor. Expected input shape is (batch, ..., var, level, lat, lon)
            preds: Tensor. Expected input shape is (batch, ..., var, level, lat, lon)
        Returns:
            None
        """

        self.nsamples += preds.shape[0]

        # for auto-broadcast
        self.mse = self.mse + self.wmse(targets, preds).sum(0)
        self.rmse_before_time_avg = self.rmse_before_time_avg + self.wmse(
                targets, preds
                ).sqrt().sum(0)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute final metrics utilizing internal states.
        Returns:
            Dict: mapping metric name to tensor holding computed metric.
                  holds one tensor per variable and metric pair ie. mse_wind_speed.
        """
        all_metrics = dict(
            rmse_before_time_avg=self.rmse_before_time_avg / self.nsamples,
            mse=self.mse / self.nsamples,
            rmse=(self.mse / self.nsamples).sqrt(),
        )
        return all_metrics



class DeterministicSEEPS(Metric, MetricBase):
    def __init__(
        self,
        data_shape: tuple,
        compute_lat_weights_fn: Callable[[int], torch.tensor] = compute_lat_weights_weatherbench,
        parent_loss_instance = None,
    ):
        """
        Args:
            variable_indices: Mapping from variable name to (var, lev) index into tensor holding computed metric.
                ie. dict(T2m=(2, 0), U10=(0, 0), V10=(1, 0), SP=(3, 0)).
                Will index into metric tensor with (..., *index) to handle extra dimensions such as multistep.
            compute_lat_weights_fn: Function to compute latitude weights given latitude shape.
                Used for error and variance calculations. Expected shape of weights: [..., lat, 1].
                See function example in metric_base.MetricBase.
                Default function assumes latitudes are ordered -90 to 90.
        """
        Metric.__init__(self)
        MetricBase.__init__(
            self,
            compute_lat_weights_fn=compute_lat_weights_fn,
            # variable_indices=variable_indices,
            # lead_time_hours=lead_time_hours,
            # rollout_iterations=rollout_iterations,
        )
        self.parent_loss_instance = parent_loss_instance
        self.add_state("nsamples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("seeps", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, targets, preds,  timestamp):
        """Update the metric state with new predictions and targets."""
        # Update internal state with new data
        self.nsamples += preds.shape[0]
        # for auto-broadcast
        steps = targets.shape[1]
        timestamp = torch.cat([
            timestamp
            + torch.arange(0, steps, device=self.device, dtype=torch.int) * 86400 
            for timestamp in timestamp])
        
        doy =[(datetime.fromtimestamp(val).timetuple().tm_yday +1)%365 for val in timestamp.cpu().numpy()]
        
        gt = targets[:,:,4,:,:,:]#bs, time, variables, atmosphere=1, 121, 240
        fc = preds[:,:,4,:,:,:]
        self.seeps = self.seeps + self.parent_loss_instance.seeps_score(fc, gt, doy, multi_day=True)
    
    def compute(self):
        all_metrics = dict(
            SEEPS=self.seeps / self.nsamples,
        )
        return all_metrics



level_variables = [
                "geopotential",
                "u_component_of_wind",
                "v_component_of_wind",
                "temperature",
                "specific_humidity",
                "vertical_velocity"]

surface_variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
]

surface_variables_short = {
    "10m_u_component_of_wind": "U10m",
    "10m_v_component_of_wind": "V10m",
    "2m_temperature": "T2m",
    "mean_sea_level_pressure": "SP",
    "total_precipitation_24hr":"precip24h",
}

pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

level_variables_short = {
    "geopotential": "Z",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "temperature": "T",
    "specific_humidity": "Q",
    "vertical_velocity": "W",
}


def get_surface_variable_indices(variables=None):
    if variables is None:
        variables = surface_variables
    else:
        variables=variables['surface']
    """Mapping from surface variable name to (var, lev) index in ERA5 dataset."""
    return {surface_variables_short[var]: (i, 0) for i, var in enumerate(variables)}

def get_level_variable_indices(pressure_levels=pressure_levels, variables=level_variables):
    """Mapping from level variable name to (var, lev) index in ERA5 dataset."""
    out = {}
    for var_idx, var in enumerate(variables):
        var_short = level_variables_short[var]
        for lev_idx, lev in enumerate(pressure_levels):
            out[f"{var_short}{lev}"] = (var_idx, lev_idx)
    return out


def get_headline_level_variable_indices(
    pressure_levels=pressure_levels, arg_level_variables=None
):
    if arg_level_variables is None:
        arg_level_variables=level_variables
    else:
        arg_level_variables=arg_level_variables['level']
    """Mapping for main level variables."""
    out = get_level_variable_indices(pressure_levels, arg_level_variables)
    return {k: v for k, v in out.items() if k in ("Z500", "T850", "Q700", "U850", "V850")}




class Era5DeterministicMetrics(TensorDictMetricBase):
    """Wrapper class around EnsembleMetrics for computing over surface and level variables.

    Handles batches coming from Era5 Dataloader.

    Accepted tensor shapes:
        targets: (batch, ..., timedelta, var, level, lat, lon)
        preds: (batch, nmembers, ..., timedelta, var, level, lat, lon)

    Return dictionary of metrics reduced over batch, lat, lon.
    """
    def __init__(
        self,
        compute_lat_weights_fn: Callable[[int], torch.tensor] = compute_lat_weights_weatherbench,
        pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], 
        #pressure_levels=era5.pressure_le
        #num_level_variables=len(level_variables),
        #num_level_variables=len(era5.level_variables),
        lead_time_hours: int = 24,
        rollout_iterations: int = 1,
        variables=None,
        parent_loss_instance=None,
    ):
        """
        Args:
            pressure_levels: pressure levels in data (used to get `variable_indices`).
            level_data_shape: (var, lev) shape for level variables.
            num_level_variables: Number of level variables (used to compute data_shape).
            rollout_iterations: Number of rollout iterations in multistep predictions.
                this option labels each timestep separately in output metric dict.
                Assumes that data shape of predictions/targets are [batch, ..., multistep, var, lev, lat, lon]


        """
        num_level_variables = len(level_variables) if variables is None else len(variables['level'])
        #global surface_variables
        surface_variables = globals()['surface_variables'] if variables is None else variables['surface']

        metrics_dict = {
                'surface':LabelDictWrapper(
                    DeterministicRMSE(
                        data_shape=(len(surface_variables), 1),
                        compute_lat_weights_fn=compute_lat_weights_fn,
                    ),
                    variable_indices=add_timedelta_index(
                        get_surface_variable_indices(variables),
                        lead_time_hours=lead_time_hours,
                        rollout_iterations=rollout_iterations,
                    ),
                ),
                'level':LabelDictWrapper(
                    DeterministicRMSE(
                        data_shape=(num_level_variables, len(pressure_levels)),
                        compute_lat_weights_fn=compute_lat_weights_fn,
                    ),
                    variable_indices=add_timedelta_index(
                        get_headline_level_variable_indices(pressure_levels, variables),
                        lead_time_hours=lead_time_hours,
                        rollout_iterations=rollout_iterations,
                    ),
                ),
                }
        if variables is not None:
            if "total_precipitation_24hr" in variables['surface']:
                metrics_dict['surface_seeps'] =LabelDictWrapper(
                        DeterministicSEEPS(
                            data_shape=(1, 1),
                            compute_lat_weights_fn=compute_lat_weights_fn,
                            parent_loss_instance=parent_loss_instance,
                        ),
                    variable_indices=add_timedelta_index(
                        get_surface_variable_indices(dict(surface=['total_precipitation_24hr'])),
                        lead_time_hours=lead_time_hours,
                        rollout_iterations=rollout_iterations,
                    ),
                )


        super().__init__(
                **metrics_dict
        )

