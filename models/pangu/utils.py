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
import pdb



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




class Loss_class():
    def __init__(self, device, multistep, lead_time_hours=24, precision=torch.float32, variables=None):
        # define coeffs for loss
        self.thresholds = torch.load('seeps_thresholds_240.pt', map_location=device)
        self.con_table = torch.load('seeps_contingency_table_240.pt', map_location=device)
        user_weatherbench_lat_coeff = True
        compute_weights_fn = self.compute_lat_weights_weatherbench
        # compute_weights_fn = (
        #     compute_lat_weights_weatherbench
        #     if use_weatherbench_lat_coeffs
        #     else compute_lat_weights
        # )
        area_weights = compute_weights_fn(121)

        pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        #era5 is from geoarches.dataloaders import era5
        pressure_levels = torch.tensor(pressure_levels).to(precision)
        #pressure_levels = torch.tensor(era5.pressure_levels).float()
        vertical_coeffs = (pressure_levels / pressure_levels.mean()).reshape(-1, 1, 1)

        # define relative surface and level weights
        total_coeff = 10
        surface_coeffs = 4 * torch.tensor([1.0, 1.0, 1.0, 1.0]).reshape(
            -1, 1, 1, 1
        )  # graphcast, mul 4 because we do a mean
        level_coeffs = 6 * torch.tensor(1).reshape(-1, 1, 1, 1)
        self.loss_coeffs = TensorDict(
            surface=area_weights * surface_coeffs / total_coeff,
            level=area_weights * level_coeffs * vertical_coeffs / total_coeff,
        )#.to(precision)

        # if loss_delta_normalization:#True 
        if variables is None:#Default
            geoarches_stats_path= os.path.join(os.path.dirname(__file__), '../../data/era_from_arches/')
            # assumes include vertical wind component
            pangu_stats = torch.load(
                geoarches_stats_path + "/pangu_norm_stats2_with_w.pt", weights_only=True
            )
            '''
            # mul by first to remove norm, div by second to apply fake delta normalization
            self.loss_delta_scaler = TensorDict(
                level=pangu_stats["level_std"]
                / torch.tensor(
                    [5.9786e02, 7.4878e00, 8.9492e00, 2.7132e00, 9.5222e-04, 0.3]
                ).reshape(-1, 1, 1, 1),
                surface=pangu_stats["surface_std"]
                / torch.tensor([3.8920, 4.5422, 2.0727, 584.0980]).reshape(-1, 1, 1, 1),
            )
            '''
            self.loss_delta_scaler = TensorDict(
                level=pangu_stats["level_std"],
                surface=pangu_stats["surface_std"],
            )
            # TODO those parameters may come from pangu_stats["level_std"][:,0,:,:]
            # and from pangu_stats["surface_std"]
            self.loss_coeffs = self.loss_coeffs * self.loss_delta_scaler.pow(2)#self.pow)

        else:#this one includes precipitation
            geoarches_stats_path= os.path.join(os.path.dirname(__file__), '../../data/era_from_arches/')
            # assumes include vertical wind component
            pangu_stats = torch.load(
                geoarches_stats_path + "/incl_precip_surface_geop.pt", weights_only=True
            )
            # mul by first to remove norm, div by second to apply fake delta normalization
            self.loss_delta_scaler = TensorDict(
                level=pangu_stats["level_std"],
                surface=pangu_stats["surface_std"],
            )
            #self.loss_coeffs = self.loss_coeffs * self.loss_delta_scaler.pow(2)#self.pow)

        self.loss_coeffs = self.loss_coeffs.to(device)

        # This could be a ModuleDict as well!
        self.test_metrics = nn.ModuleList([Era5DeterministicMetrics(lead_time_hours = lead_time_hours, rollout_iterations=multistep, variables=variables, parent_loss_instance=self)]).to(device)
        self.device=device
        # for two losses - initial values
        self.mse_scale = 35
        self.physics_scale = 0.0002


    def loss(self, pred, gt, denorm, multistep=False,slice_dim=5-1, physics_loss_coeff=0, timestamp=None):
        '''
        if multistep:  # means we have to compute multistep loss
            # discount for multistep loss
            lead_iter = next(iter(gt.values())).shape[1]
            future_coeffs = (
                torch.tensor([1 / (1 + i) ** 2 for i in range(lead_iter)])
                .to(self.device)
                .reshape(-1, 1, 1, 1, 1)
            )

            self.loss_coeffs.apply(lambda x: x * future_coeffs)
        '''
        if slice_dim is not None:
            # RMSE slice: everything else INCLUDING level data - as TensorDict
            gt_rmse = TensorDict({
                'surface': gt['surface'][:, :slice_dim, :, :, :],  
                'level': gt['level'],  
            }, batch_size=gt['surface'].shape[0])  # Set batch size
            
            pred_rmse = TensorDict({
                'surface': pred['surface'][:, :slice_dim, :, :, :],
                'level': pred['level'],
            }, batch_size=pred['surface'].shape[0])
            weighted_error = (pred_rmse - gt_rmse).abs().pow(2)#.mul(self.loss_coeffs.to(gt.dtype))
            loss = sum(weighted_error.mean().values())

            #gt_precip = gt['surface'][:, :, slice_dim, :, :]
            gt_precip = denorm(gt)['surface'][..., slice_dim, :, :,:]
            pred_precip = denorm(pred)['surface'][..., slice_dim, :, :, :]
            #print([datetime.fromtimestamp(val) for val in timestamp.cpu().numpy()])
            day_of_years=[(datetime.fromtimestamp(val).timetuple().tm_yday + 1)%366 for val in timestamp.cpu().numpy()]
            seeps_score = self.seeps_score(pred_precip, gt_precip, day_of_years, multi_day=multistep)
            return loss, seeps_score
        else:
            weighted_error = (pred- gt).abs().pow(2)#.mul(self.loss_coeffs.to(gt.dtype))
            loss = sum(weighted_error.mean().values())
        pdb.set_trace()
        weighted_error = (pred - gt).abs().pow(2)#.mul(self.loss_coeffs.to(gt.dtype))
        loss = sum(weighted_error.mean().values())

        
        if physics_loss_coeff != 0:
            extra_loss = self.pde_loss(pred, gt)
            with torch.no_grad():
                self.mse_scale = 0.99 * self.mse_scale + 0.01 * loss.detach()
                self.physics_scale= 0.90 * self.physics_scale+ 0.1 * extra_loss.detach()
            loss = loss / self.mse_scale + 1e-6
            extra_loss = extra_loss / self.physics_scale + 1e-6

            loss = loss + physics_loss_coeff* extra_loss

        return loss

    def seeps_score(self, fc, gt, doy, multi_day=False):
        #NOTE TODO: THE NAMES ARE WRONG! FIRST argument is forecast, second is ground truth
        # but just the names. rest is fine.
        '''
        Args:
        '''
        gt_prob = self.differentiable_threshold_one_hot(gt, doy, temperature=0.0001)
        fc_prob = self.differentiable_threshold_one_hot(fc, doy, temperature=0.0001)

        #pdb.set_trace()
        if torch.is_grad_enabled():
            con_table = torch.nan_to_num(self.con_table)
        else:
            con_table = self.con_table

        loss = torch.einsum('...ij,...j->...i', con_table[doy,...], gt_prob.transpose(1,2))
        loss = torch.einsum('...i,...i->...', fc_prob.transpose(1,2), loss)
        if torch.is_grad_enabled():
            # This is a little bit of an issue since later mean() will be taken,
            # but zeros will count as existent entries (i.e. correct classified)
            loss = torch.nan_to_num(loss)
        if multi_day:
            return loss.view(2, -1, 240, 121).nanmean(dim=[0,2,3])
        loss = loss.nanmean()
        # maybe return 1-loss ? (that would mean: high = good)
        return loss

    def differentiable_threshold_one_hot(self, input_vals, doy, temperature=0.5):
        """
        Differentiable version using smooth approximations.
        Args:
            input_vals: input tensor of shape [?]
            thresholds: seeps_thresholds it is 0.25mm/day 
            thresholds: upper seeps threshold - needs to be on the same unit.
            temperature: Controls smoothness (lower = sharper transitions)
        """
        lower_thresh = self.thresholds[doy,:,:, 0]
        upper_thresh = self.thresholds[doy,:,:, 1]
        # Smooth approximations using sigmoid
        #lower threshold: 10, 240, 121 - vs 2, 240, 121
        #input vals 2, 5, 121, 240 - vs 2, 1, 121, 240
        input_vals = input_vals.reshape(
                lower_thresh.shape[0],
                lower_thresh.shape[2],
                lower_thresh.shape[1])
        
        if torch.is_grad_enabled():
            input_vals = torch.nan_to_num(input_vals)
            upper_thresh = torch.nan_to_num(upper_thresh)
            lower_thresh = torch.nan_to_num(lower_thresh)
        prob_below = torch.sigmoid((lower_thresh.transpose(1,2) - input_vals) / temperature)
        prob_above = torch.sigmoid((input_vals - upper_thresh.transpose(1,2)) / temperature)# THIS creates NaNs!

        prob_between = 1 - prob_below - prob_above
        
        # Stack probabilities to create soft one-hot encoding
        output = torch.stack([prob_below, prob_between, prob_above], dim=-1)
        
        if torch.is_grad_enabled():
            # This is a little bit of an issue since later mean() will be taken,
            # but zeros will count as existent entries (i.e. correct classified)
            output = torch.nan_to_num(output)
        return output

    def pde_loss(self, pred, gt):
        # utils.py you can find the variables
        # level_variables = [
        #"geopotential",
        #"u_component_of_wind",
        #"v_component_of_wind",
        #"temperature",
        #"specific_humidity",
        #"vertical_velocity"]
        # Constants
        geopotential = pred['level'][:,0,:,:,:]# batch, variable, pressure_level, width, height
        temperature = pred['level'][:,3,:,:,:]# batch, variable, pressure_level, width, height
        wind_u = pred['level'][:,1,:,:,:]# batch, variable, pressure_level, width, height
        wind_v = pred['level'][:,2,:,:,:]# batch, variable, pressure_level, width, height

        loss = self.hydrostatic_loss(geopotential, temperature)
        #loss = self.geostrophic_balance_loss(wind_u, wind_v, geopotential, device=self.device)
        return loss

    def hydrostatic_loss(self, geopotential, temperature):
        R = 287.0  # Specific gas constant for dry air (J/kg/K)
        pressure_levels_hPa = torch.tensor([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]).to(self.device)
        pressure_levels_Pa = pressure_levels_hPa * 100  # Convert hPa to Pa

        dPhi = torch.gradient(geopotential, dim=1)[0]  # ∂Φ between pressure levels
        dp = torch.gradient(pressure_levels_Pa)[0].view(1, -1, 1, 1).to(geopotential.dtype)

        dPhi_dp = dPhi / dp  # ∂Φ/∂p
        
        p_broadcast = pressure_levels_Pa.view(1, -1, 1, 1)  # [1, pressure_levels, 1, 1]
        hydrostatic_theory = -R * temperature / p_broadcast

        loss = F.mse_loss(dPhi_dp, hydrostatic_theory)
        return loss


    def geostrophic_balance_loss(self,wind_u, wind_v, geopotential, device='cuda'):
        """
        PINN loss for geostrophic balance with height levels.
        
        Args:
            wind_u: [batch, height_levels, lat, lon] - zonal wind component (m/s)
            wind_v: [batch, height_levels, lat, lon] - meridional wind component (m/s) 
            geopotential: [batch, height_levels, lat, lon] - geopotential height (m²/s²)
            
        Returns:
            loss: scalar tensor representing geostrophic balance violation
        """
        
        # Physical constants
        OMEGA = 7.2921159e-5  # Earth's angular velocity (rad/s)
        R_EARTH = 6.371e6     # Earth's radius (m)
        F_MIN = 1e-10         # Minimum Coriolis parameter to avoid division by zero
        
        batch_size, n_height, n_lat, n_lon = geopotential.shape
        
        # Create coordinate grids for 121 lat x 240 lon
        # 121 lat points: -90 to +90 degrees (full global coverage)
        # 240 lon points: 0 to 358.5 degrees
        lat_deg = torch.linspace(-90.0, 90.0, n_lat, device=device)    # 121 points
        lon_deg = torch.linspace(0.0, 358.5, n_lon, device=device)    # 240 points
        
        # Convert to radians
        lat_rad = lat_deg * np.pi / 180.0
        
        # Compute Coriolis parameter f = 2Ω sin(φ)
        f = 2 * OMEGA * torch.sin(lat_rad)  # [n_lat]
        # Expand to match full tensor dimensions [batch, height, lat, lon]
        f = f.view(1, 1, -1, 1).expand(batch_size, n_height, n_lat, n_lon)
        
        # Numerical stability: avoid division by zero near equator
        f_stable = torch.where(torch.abs(f) < F_MIN, 
                              torch.sign(f) * F_MIN, 
                              f)
        
        # Grid spacing in meters
        dlat = (lat_deg[1] - lat_deg[0]) * np.pi / 180.0  # Convert degree to radians
        dlat_m = dlat * R_EARTH
        
        # Longitude spacing varies with latitude (cos(lat) factor)
        dlon = (lon_deg[1] - lon_deg[0]) * np.pi / 180.0
        cos_lat = torch.cos(lat_rad).view(1, 1, -1, 1).expand(batch_size, n_height, n_lat, n_lon)
        dlon_m = dlon * R_EARTH * cos_lat
        
        # Compute spatial gradients of geopotential using torch.gradient
        # ∂Φ/∂y (northward gradient) - gradient along latitude dimension (dim=2)
        dPhi_dy = torch.gradient(geopotential, dim=2)[0] / dlat_m
        
        # ∂Φ/∂x (eastward gradient) - gradient along longitude dimension (dim=3)
        dPhi_dx = torch.gradient(geopotential, dim=3)[0] / dlon_m
        
        # Calculate theoretical geostrophic winds
        # Geostrophic balance: fu_g = -∂Φ/∂y, fv_g = +∂Φ/∂x
        u_geostrophic = -(1.0 / f_stable) * dPhi_dy  # [batch, height, lat, lon]
        v_geostrophic = +(1.0 / f_stable) * dPhi_dx  # [batch, height, lat, lon]
        
        # Apply equatorial mask (optional: reduce weight near equator)
        equatorial_mask = torch.abs(lat_rad) > (5.0 * np.pi / 180.0)  # Exclude ±5° from equator
        equatorial_weight = equatorial_mask.float().view(1, 1, -1, 1).expand_as(wind_u)
        
        # Compute residuals (difference between actual and geostrophic winds)
        residual_u = wind_u - u_geostrophic
        residual_v = wind_v - v_geostrophic
        
        # Apply equatorial weighting (optional)
        weighted_residual_u = residual_u * equatorial_weight
        weighted_residual_v = residual_v * equatorial_weight
        
        # PINN loss: MSE of residuals
        loss_u = F.mse_loss(weighted_residual_u, torch.zeros_like(weighted_residual_u))
        loss_v = F.mse_loss(weighted_residual_v, torch.zeros_like(weighted_residual_v))
        
        total_loss = loss_u + loss_v
        
        return total_loss


    def compute_lat_weights_weatherbench(self, latitude_resolution: int) -> torch.tensor:
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

    def training_step(self, pred, batch, denormalize):
        #denormalize = self.trainer.train_dataloader.dataset.denormalize # method
        #for metric in self.train_metrics:
        #    metric.reset()
        if "future_states" not in batch:
            # standard prediction
            loss, seeps_loss = self.loss(pred, batch["next_state"], denormalize, physics_loss_coeff=0, timestamp=batch['timestamp'])
        else:
            assert False#no multiple training steps?
            '''
            # multistep prediction
            lead_iter = batch["future_states"].shape[1]
            pred_future_states = self.forward_multistep(batch, iters=lead_iter)
            loss = self.loss(pred_future_states, batch["future_states"], multistep=True)

            # metrics
            rollout_iterations = self.cfg.train.metrics_kwargs.rollout_iterations
            pdb.set_trace() 
            # TODO include day-of-year and multi-day doy as well
            for metric in self.train_metrics:
                metric.update(
                    denormalize(batch["future_states"][:, :rollout_iterations]),
                    denormalize(pred_future_states[:, :rollout_iterations]),
                    timestamp=batch['timestamp']
                )
            '''
        return loss, seeps_loss

    def validation_step(self, pred, batch, denormalize, future_states=False):
        #if "future_states" not in batch:
        if not future_states:
            # standard prediction
            loss, seeps = self.loss(pred, batch["next_state"], denormalize, timestamp=batch['timestamp'])
        '''else:
            # multistep prediction
            loss = self.loss(pred, batch["future_states"], multistep=True)
        '''
        return loss, seeps

    def test_step(self, pred, batch, denormalize, rollout_iterations=1, future_states=False):
        timestamp=batch['timestamp']
        if not future_states:
            # standard prediction
            loss = self.loss(pred, batch["next_state"], denormalize)
            for metric in self.test_metrics:
                metric.update(
                    denormalize(batch["next_state"])[:, None],
                    denormalize(pred)[:, None],
                    timestamp=batch['timestamp']
                )
        else:
            steps = batch["future_states"].shape[1]
            timestamp_short_var = torch.cat([
                timestamp
                + torch.arange(0, steps, device=self.device, dtype=torch.int) * 86400 
                for timestamp in batch["timestamp"]])
            loss, seeps = self.loss(pred, batch["future_states"], denormalize, multistep=True,timestamp=timestamp_short_var)
            # metrics
            for metric in self.test_metrics:
                metric.update(
                    denormalize(batch["future_states"][:, :rollout_iterations]),
                    denormalize(pred[:, :rollout_iterations]),
                    timestamp=batch['timestamp']
                )
        return loss, seeps





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
                # Handle returned dictionary.
                if isinstance(output, dict):
                    if len(aggregated_outputs) - 1 < i:
                        aggregated_outputs.append({})
                    if aggregated_outputs[i].keys().isdisjoint(output.keys()):
                        aggregated_outputs[i].update(output)
                    else:
                        aggregated_outputs[i].update({f"{k}_{key}": v for k, v in output.items()})
                # Handle returned xarray dataset.
                elif isinstance(output, xr.Dataset):
                    if len(aggregated_outputs) - 1 < i:
                        aggregated_outputs.append([])
                    aggregated_outputs[i].append(output)

        for output in aggregated_outputs:
            if isinstance(output, list):
                merged_dataset = xr.merge(output)
                aggregated_outputs[i] = merged_dataset

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

