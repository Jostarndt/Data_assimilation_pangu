#  uv run prio_knowledge_training_single_gpu.py --model_path pangu_weather_24.onnx --data_path /path/to/era5.nc --prior_known_dim=[1,2] --reg_param 1e8
import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import wandb
import time
from pathlib import Path
from torch.amp import GradScaler, autocast

from tensordict.tensordict import TensorDict

from onnx2torch import convert

#from torch.distributed.fsdp import fully_shard, FSDPModule
#import torch.distributed as dist
#from torch.distributed.device_mesh import init_device_mesh
#from torch.distributed.fsdp import CPUOffload

# Append project root to sys.path to allow imports from utils and other directories
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

def main():
    # Argument parser for configuration file
    parser = argparse.ArgumentParser(description='Training script for Model A')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to Pangu-Weather ONNX model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to ERA5 NetCDF file')
    parser.add_argument('--name', type=str, default=None, help='Experiment name for wandb')
    parser.add_argument('--checkpoint', help='Path to checkpoint')
    parser.add_argument('--git_commit', type=str, help='Git commit hash')
    parser.add_argument('--prior_known_dim', type=str, default=None, help='List of integers, e.g., "[1,2,3]"')
    parser.add_argument('--known_region', action='store_true', default=False, help='Set to True if region is known')
    parser.add_argument('--known_atmosphere',type=lambda x: None if x.lower() == 'none' else int(x),default=None)
    parser.add_argument('--slurm_id', type=str, default=None, help='SLURM Job ID')
    parser.add_argument('--reg_param', type=float, default=1e8, help='regularization parameter')
    parser.add_argument('--LBFGSsteps', type=int, default=15)
    parser.add_argument('--r_tensors_path', type=str, default=None, help='Path to r_tensors.pt for blended rollout')

    args = parser.parse_args()
    print("SLURM job id: ", args.slurm_id)

    if args.prior_known_dim is not None:
        import ast
        try:
            print(args.prior_known_dim)
            prior_known_dim = ast.literal_eval(args.prior_known_dim.strip("'\""))
            if not isinstance(prior_known_dim, list):
                raise ValueError("Must be a list")
        except (ValueError, SyntaxError):
            raise ValueError("prior_known_dim must be a valid list format like '[1,2,3]'")
    else:
        prior_known_dim = None
    
    # Load configuration from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config['git_commit'] = args.git_commit

    config['reg_param']=args.reg_param
    config['prior_known_dim']=prior_known_dim
    config['known_region']=args.known_region
    config['known_atmosphere']=args.known_atmosphere
    config['LBFGSsteps']=args.LBFGSsteps


    wandb.init(project=config['wandb']['project'], config=config, name=args.name,
               tags=[args.slurm_id] if args.slurm_id else [])
    wandb.save("prio_knowledge_training_single_gpu.py")

    batch_size=1

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = 'cuda:0'
    print(f'Using device: {device}')
    
    current_file_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_file_dir.parent.parent / "data/era5_dataloader"))
    from era5 import Era5Forecast
    
    variables = {
            'surface':[
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature"],
            'level':[
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind"]
            }

    train_dataset = Era5Forecast(domain="train", lead_time_hours=24,
            multistep=1,
            variables=variables, norm_scheme=False,
            path=args.data_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=16, collate_fn=collate_fn)

    print("Loading model")
    precision = torch.float

    pytorch_model = convert(args.model_path)
    model = pytorch_model.to(device)
       
    # Define loss function and optimizer
    criterion = Loss_class(device, multistep = 1, lead_time_hours=24, precision=precision, variables=variables)

    scaler = GradScaler("cuda")

    print("Start training")

    train_fkt(model, train_loader, criterion, scaler, train_dataset.denormalize,
            device, precision, config, epoch=1, start_step=0, prior_known_dim=prior_known_dim, reg_param=args.reg_param,
            known_region=args.known_region, known_atmosphere=args.known_atmosphere, LBFGS_steps=args.LBFGSsteps,
            r_tensors_path=args.r_tensors_path)


def train_fkt(model, train_loader, criterion, scaler, denorm, device, precision, config, epoch,
        start_step=0, prior_known_dim=None, reg_param=1e8,
        known_region=False, known_atmosphere=None, LBFGS_steps=15, r_tensors_path=None):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    start_time=time.time()
    print("started training in train_fkt. start_step: ", start_step)
    print("The train loader has a length of :", len(train_loader))
    rel_dif_level = []
    rel_dif_surface = []
    for i, batch in enumerate(train_loader):
        print(f'sample: {i} ', flush=True)
        if i >= 15:
            break
        pop = batch.pop('prev_state')

        batch = send_to_device(batch, device, precision)

        '''
        #pdb.set_trace()
        with torch.no_grad():
            var_level, var_surface =  model(input_1 = batch['state']['level'], input_2=batch['state']['surface'])
            batch['state']['level'] = var_level.unsqueeze(0).clone()
            batch['state']['surface'] = var_surface.unsqueeze(0).unsqueeze(2).clone()

            var_level, var_surface =  model(input_1 = batch['next_state']['level'], input_2=batch['next_state']['surface'])
            batch['next_state']['level'] = var_level.unsqueeze(0).clone()
            batch['next_state']['surface'] = var_surface.unsqueeze(0).unsqueeze(2).clone()
        '''

        original_state = batch['state'].detach().clone()
        '''
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("forward pass")
        outputs_level, outputs_surface = model(input_1 = batch['state']['level'], input_2=batch['state']['surface'])#this is not normalized
        print('Forward pass requires ', torch.cuda.max_memory_allocated() / 1024**3, 'GB')
        '''

        #to scale them with their stds:
        # for mean_sea_level_pressure, 10m_u_component_of_wind, 10m_v_component_of_wind, 2m_temperature
        
        batch['state'].requires_grad_(True)

        optimizer = optim.LBFGS([batch['state']['level'],
            batch['state']['surface']],lr=1, max_iter=LBFGS_steps,

            tolerance_grad=1e-12,
            tolerance_change=1e-15, 
            history_size=18,#10 requires 30.54GB
            line_search_fn='strong_wolfe')#'strong_wolfe' or None

        # dict_keys(['timestamp', 'state', 'lead_time_hours', 'next_state', 'prev_state'])
        
        def closure():
            print(f"Reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
            optimizer.zero_grad()
            # TODO s:
            # 1 normalization
            # 2. 0/360 or -180/180 
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                #pdb.set_trace()
                outputs_level, outputs_surface = model(input_1 = batch['state']['level'], input_2=batch['state']['surface'])
            outputs = {'level':outputs_level, 'surface':outputs_surface}
            loss = criterion.training_step(pred=outputs, batch=batch, prior_known_dim=prior_known_dim, complement=False,
                    known_region=known_region, known_atmosphere=known_atmosphere)

            std_tensor_surface = torch.tensor([3.35e4, 4.44, 3.83, 92.4], device=loss.device,
                    dtype=loss.dtype, requires_grad=False).unsqueeze(1).unsqueeze(2)
            std_tensor_level = torch.tensor([2.625e4, 0.002213, 36.25, 7.481, 4.745], device=loss.device,
                    dtype=loss.dtype, requires_grad=False).unsqueeze(1).unsqueeze(2).unsqueeze(3)

            reg_term_surface = ((batch['state']['surface'] - original_state['surface'].detach())/std_tensor_surface).abs().pow(2).mean()
            reg_term_level = ((batch['state']['level'] - original_state['level'].detach())/std_tensor_level).abs().pow(2).mean()

            reg_term = reg_param*(reg_term_surface+reg_term_level)
            #reg_term = 1e13*sum((batch['state'][k] - original_state[k].detach()).pow(2).mean() for k in original_state.keys())

            print(f"forward-with_gradients successfull. Loss:{loss}, regularization term {reg_term}")
            loss = loss + reg_term
            loss.backward()
            return loss

			
        for step in range(1):
            loss = optimizer.step(closure)
            print("-"*20)
            torch.cuda.empty_cache()
            with torch.no_grad():
                outputs_level, outputs_surface = model(input_1 = batch['state']['level'], input_2=batch['state']['surface'])
                outputs = {'level':outputs_level, 'surface':outputs_surface}

                loss = criterion.training_step(pred=outputs, batch=batch, 
                        prior_known_dim=None, complement=False, known_region=False, known_atmosphere=None)

                prior_loss = criterion.training_step(pred=outputs, batch=batch, 
                        prior_known_dim=prior_known_dim, complement=False, known_region=known_region, known_atmosphere=known_atmosphere)

                reg_term = sum((batch['state'][k] - original_state[k].detach()).abs().pow(2).mean() for k in original_state.keys())
                #print(f'Overall forecast loss {loss}, Regularization term: {reg_term}, Loss only on prior_known: {prior_loss}, optimizing_loss: {prior_loss + 10*reg_term}')
                print(f'Overall forecast loss {loss}, Loss only on prior_known: {prior_loss}, Regularization term: {reg_term},  optimizing_loss: {prior_loss + reg_param*reg_term}')

                orig_batch = batch.copy()
                orig_batch['state'] = original_state
                outputs_level_orig, outputs_surface_orig = model(input_1 = orig_batch['state']['level'], input_2=orig_batch['state']['surface'])
                outputs_orig = {'level':outputs_level_orig, 'surface':outputs_surface_orig}

                loss_orig = criterion.training_step(pred=outputs_orig, batch=orig_batch,
                        prior_known_dim=None, complement=False, known_region=False, known_atmosphere=None)
                
                improvement = ((loss / loss_orig) - 1) * 100 # improvement in percent. negative number means improvement
                
                complement_loss_orig = criterion.training_step(pred=outputs_orig, batch=orig_batch,
                        prior_known_dim=prior_known_dim, complement=True, known_region=known_region, known_atmosphere=known_atmosphere)

                complement_loss_update = criterion.training_step(pred=outputs, batch=batch,
                        prior_known_dim=prior_known_dim, complement=True, known_region=known_region, known_atmosphere=known_atmosphere)

                alternative_improvement = ((complement_loss_update / complement_loss_orig) -1 )*100
                #TODO maybe the mean of the data should be added here for change_pct
                #
                change_pct = 100 * sum((batch['state'][k] - original_state[k]).norm() / (original_state[k].norm() + 1e-8) for k in original_state.keys()) / len(original_state.keys())

                # mean_tensor_level = torch.tensor([2.628e+03, 0.0003133, 5.132, 1.023, 0.04966], device=loss.device,
                #         dtype=loss.dtype, requires_grad=False)
                std_tensor_level = torch.tensor([2.625e4, 0.002213, 36.25, 7.481, 4.745], device=loss.device,
                        dtype=loss.dtype, requires_grad=False).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                mse_orig = ((outputs_orig['level']-orig_batch['next_state']['level'])/ std_tensor_level).pow(2).mean(axis=[0,2,3,4]).squeeze()
                mse_new = ((outputs['level']-batch['next_state']['level'])/std_tensor_level).pow(2).mean(axis=[0,2,3,4])
                relative_diff = (mse_new - mse_orig) / (mse_orig)#+mean_tensor_level)
                print(relative_diff.mean(), " mean of level relative diffs", relative_diff)
                rel_dif_level.append(relative_diff.cpu().numpy())
                mean_tensor_surface = torch.tensor([1.7e4, -0.077, -0.14, 66.72], device=loss.device,
                        dtype=loss.dtype, requires_grad=False)
                std_tensor_surface = torch.tensor([3.35e4, 4.44, 3.83, 92.4], device=loss.device,
                        dtype=loss.dtype, requires_grad=False).unsqueeze(1).unsqueeze(2)
                
                #mean_tensor_surface = torch.tensor([1.7e4, -0.077, -0.14, 66.72], device=loss.device,
                #        dtype=loss.dtype, requires_grad=False)
                #std_tensor_surface = torch.tensor([3.35e4, 4.44, 3.83, 92.4], device=loss.device,
                #        dtype=loss.dtype, requires_grad=False).unsqueeze(1).unsqueeze(2)

                mse_orig = (outputs_orig['surface'].squeeze() - orig_batch['next_state']['surface'].squeeze()).pow(2).mean(axis=[1,2])
                mse_new = (outputs['surface'].squeeze() - batch['next_state']['surface'].squeeze()).pow(2).mean(axis=[1,2])

                #this +mean_tensor_surface should not be there! this is wrong. also later its wrong.
                relative_diff = (mse_new - mse_orig) / (mse_orig)
                print(relative_diff.mean(), " mean of surface relative diffs", relative_diff)
                rel_dif_surface.append(relative_diff.cpu().numpy())
                print(f"Change in state (unnormalized): {change_pct}%")
                print(f"Full improvement: {improvement}%")
                print(f"Improvement only on complement-metric: {alternative_improvement}%")
                #pdb.set_trace()
                #print('next iteration')

                if 'future_states' in batch:
                    rollout_orig = []
                    rollout_corr = []

                    outputs_next_corrected = outputs
                    outputs_orig_next = outputs_orig

                    if r_tensors_path:
                        rollout_blended = []
                        r_tensors = torch.load(r_tensors_path)
                        outputs_next_blended = {
                            'level': outputs_orig['level'] + r_tensors['r_level'].unsqueeze(-1).unsqueeze(-1) * (outputs['level'] - outputs_orig['level']),
                            'surface': outputs_orig['surface'] + r_tensors['r_surface'].unsqueeze(-1).unsqueeze(-1) * (outputs['surface'] - outputs_orig['surface'])}
                        rollout_blended.append(outputs_next_blended)

                    for rstep in range(batch['future_states'].shape[-1]):
                        outputs_level_orig, outputs_surface_orig = model(input_1 = outputs_orig_next['level'], input_2 = outputs_orig_next['surface'])
                        outputs_orig_next = {'level':outputs_level_orig, 'surface':outputs_surface_orig}
                        rollout_orig.append(outputs_orig_next)

                        outputs_level, outputs_surface = model(input_1 = outputs_next_corrected['level'], input_2=outputs_next_corrected['surface'])
                        outputs_next_corrected = {'level':outputs_level, 'surface':outputs_surface}
                        rollout_corr.append(outputs_next_corrected)

                        if r_tensors_path:
                            outputs_level, outputs_surface = model(input_1 = outputs_next_blended['level'], input_2=outputs_next_blended['surface'])
                            outputs_next_blended = {'level':outputs_level, 'surface':outputs_surface}
                            rollout_blended.append(outputs_next_blended)

                    rollout_orig_stacked = {
                            'level': torch.stack([r['level'].cpu() for r in rollout_orig]),
                            'surface': torch.stack([r['surface'].cpu() for r in rollout_orig])
                            }
                    rollout_corr_stacked = {
                            'level': torch.stack([r['level'].cpu() for r in rollout_corr]),
                            'surface': torch.stack([r['surface'].cpu() for r in rollout_corr])
                            }
                    if r_tensors_path:
                        rollout_blend_stacked = {
                                'level': torch.stack([r['level'].cpu() for r in rollout_blended]),
                                'surface': torch.stack([r['surface'].cpu() for r in rollout_blended])
                                }

        save_dict = {
            'corrected state': batch['state'],
            'original state': original_state,
            'original_forecast': outputs_orig,
            'corrected_forecast': outputs,
            'target': batch['next_state'],
        }
        if 'future_states' in batch:
            save_dict['original_rollout'] = rollout_orig_stacked
            save_dict['corrected_rollout'] = rollout_corr_stacked
            if r_tensors_path:
                save_dict['blended_rollout'] = rollout_blend_stacked
            save_dict['gt_rollout'] = batch['future_states']

        torch.save(save_dict,
            os.path.join(config['output']['model_save_path'], f'correction_tensordicts{i}.pt'))

        end_time = time.time()
        time_diff = end_time - start_time
        wandb.log({
            'step': i,
            'overall improvement': improvement,
            'complementary Improvement':alternative_improvement,
            'Change in state': change_pct.item()
        })
        start_time=time.time()



def collate_fn(lst):
    return {k: torch.stack([x[k] for x in lst]) for k in lst[0]}




def send_to_device(batch, device, precision=None):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key == "timestamp":
                batch[key] = value.to(device)
            else:
                batch[key] = value.to(precision).to(device)
        elif isinstance(value, TensorDict):
            batch[key] = value.to(precision).to(device)  
    return batch


class Loss_class():
    def __init__(self, device, multistep, lead_time_hours=24, precision=torch.float32, variables=None):
        # define coeffs for loss
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

        self.loss_coeffs = self.loss_coeffs.to(device)
        self.device=device
        torch.manual_seed(42)
        self.rand_mask = torch.rand(721, 1440)
        torch.save(self.rand_mask, "rand_mask.pt")


    def loss(self, pred, gt, multistep=False, prior_known_dim=None, original_state=None, physics_loss_coeff=0, timestamp=None,
            known_region=False, known_atmosphere=None, complement=False):
        #Different cases
        if known_atmosphere is not None:
            #pdb.set_trace()
            mask = torch.zeros((721, 1440), dtype=torch.bool)
            if known_atmosphere == 1:
                mask = ~mask
            elif known_atmosphere == 2:
                mask[::2,::2]=True
                mask = self.rand_mask < 0.25
            elif known_atmosphere == 3:
                mask[::3,::3]=True
            elif known_atmosphere == 4:
                mask[240:481, :] = True#tropics
            elif known_atmosphere == 5:#only upper-air atmosphere
                mask = ~mask
            elif known_atmosphere == 6:
                mask[::4,::4]=True
            elif known_atmosphere == 7:
                mask = self.rand_mask
            elif known_atmosphere == 8:
                mask = self.rand_mask < 0.25
            elif known_atmosphere == 9:
                mask = self.rand_mask < 0.11
            elif known_atmosphere == 10:
                mask = self.rand_mask < 0.0625

            ttype=gt['surface'].dtype
            tdevice=gt['surface'].device

            std_tensor_surface = torch.tensor([3.35e4, 4.44, 3.83, 92.4], device=tdevice,
                    dtype=ttype, requires_grad=False).unsqueeze(1).unsqueeze(2) 
            loss_means_surface = torch.tensor([3.3740e-04, 6.2160e-01, 7.8206e-01, 1.5008e-03], device=tdevice,
                    dtype=ttype, requires_grad=False)

            '''
            std_tensor_surface = self.loss_delta_scaler['surface']
            std_tensor_surface.requires_grad = False
            std_tensor_level = self.loss_delta_scaler['level']
            std_tensor_level.requires_grad = False
            
            loss_means_path = os.path.join(os.path.dirname(__file__), 'error_stats.pt')
            loss_means_level = torch.load(loss_means_path)['level']
            loss_means_surface = torch.load(loss_means_path)['surface']
            
            '''
            std_tensor_level = torch.tensor([2.625e4, 0.002213, 36.25, 7.481, 4.745],
                    device= tdevice,
                    dtype= ttype,
                    requires_grad=False).unsqueeze(1).unsqueeze(2).unsqueeze(3) 
            loss_means_level = torch.tensor(
                    [2.1723e-04, 8.4831e-02, 2.1656e-03, 2.5990e-01, 6.2383e-01],
                    device= tdevice,
                    dtype= ttype,
                    requires_grad=False)

            if complement:
                mask = ~mask

            if known_atmosphere == 5:#only upper-air atmosphere
                mask = ~mask
            #pdb.set_trace()
            loss = ((gt['surface'].squeeze()-pred['surface']).abs()/std_tensor_surface).pow(2)/loss_means_surface[:, None, None]
            loss = loss[:,mask].nanmean()
            
            if known_atmosphere == 5:#only upper-air atmosphere
                mask = ~mask

            level_loss = ((gt['level'][0,...]-pred['level']).abs()/std_tensor_level).pow(2)/loss_means_level[:, None, None, None]#.mean(axis=[1,2,3])
            level_loss = level_loss[:,:, mask].nanmean()

            loss = torch.nanmean(torch.stack([loss, level_loss]))
            return loss


        elif known_region:
            selected_pred = pred['surface'].squeeze()
            tdevice = pred['surface'].squeeze().device
            tdtype = pred['surface'].squeeze().dtype
            
            std_tensor = torch.tensor([3.35e4, 4.44, 3.83, 92.4], device=selected_pred.device,
                    dtype=selected_pred.dtype, requires_grad=False).unsqueeze(1).unsqueeze(2) 
            loss_means_surface = torch.tensor([3.3740e-04, 6.2160e-01, 7.8206e-01, 1.5008e-03], device=selected_pred.device,
                    dtype=selected_pred.dtype, requires_grad=False)

            loss = ((gt['surface'].squeeze() - pred['surface'].squeeze()).abs()/std_tensor).pow(2)#.mean(axis=[1,2])
            loss=loss/loss_means_surface[:,None,None]

            mask_variables = torch.zeros(gt['surface'].shape[1], dtype=torch.bool)
            mask_h = torch.zeros(gt['surface'].shape[-2], dtype=torch.bool)
            mask_w = torch.zeros(gt['surface'].shape[-1], dtype=torch.bool)

            mask_variables[prior_known_dim] = True
            mask_h[::2] = True#0 to 721
            mask_w[::2] = True#1440

            full_mask = (mask_variables[:, None, None] 
                    & mask_h[None, :, None] 
                    & mask_w[None, None, :])

            if complement:
                full_mask = ~full_mask

            loss  = loss[full_mask].mean()

            if complement:
                '''
                std_tensor = torch.tensor([2.625e4, 0.002213, 36.25, 7.481, 4.745],
                        device=selected_pred.device,
                        dtype=selected_pred.dtype,
                        requires_grad=False).unsqueeze(1).unsqueeze(2).unsqueeze(3) 

                loss_means_level = torch.tensor(
                        [2.1723e-04, 8.4831e-02, 2.1656e-03, 2.5990e-01, 6.2383e-01],
                        device=selected_pred.device,
                        dtype=selected_pred.dtype,
                        requires_grad=False)
                '''

                level_loss = ((gt['level'][0,...]-pred['level']).abs()/std_tensor).pow(2).mean(axis=[1,2,3])
                level_loss = (level_loss /loss_means_level).mean()

                loss = torch.nanmean(torch.stack([loss, level_loss]))
            return loss

        elif prior_known_dim is not None:
            # Create boolean mask
            mask = torch.zeros(gt['surface'].shape[1], dtype=torch.bool)
            mask[prior_known_dim] = True

            if complement:
                mask = ~mask
            # Select indices
            selected_gt = gt['surface'][0,mask, 0,:,:] 
            selected_pred = pred['surface'][mask,:,:]
            
            #std of the values
            std_tensor = torch.tensor([3.35e4, 4.44, 3.83, 92.4], device=selected_pred.device,
                    dtype=selected_pred.dtype, requires_grad=False)[mask].unsqueeze(1).unsqueeze(2)
            # mean_tensor = torch.tensor([1.7e4, -0.077, -0.14, 66.72], device=selected_pred.device,
            # dtype=selected_pred.dtype, requires_grad=False)[mask].unsqueeze(1).unsqueeze(2)

            loss_means_surface = torch.tensor([3.3740e-04, 6.2160e-01, 7.8206e-01, 1.5008e-03], device=selected_pred.device,
                    dtype=selected_pred.dtype, requires_grad=False)[mask]

            loss = ((selected_gt - selected_pred).abs()/std_tensor).abs().pow(2).mean(axis=[1,2])
            loss=(loss/loss_means_surface).mean()

            if complement:
                std_tensor = torch.tensor([2.625e4, 0.002213, 36.25, 7.481, 4.745],
                        device=selected_pred.device,
                        dtype=selected_pred.dtype,
                        requires_grad=False).unsqueeze(1).unsqueeze(2).unsqueeze(3) 

                loss_means_level = torch.tensor(
                        [2.1723e-04, 8.4831e-02, 2.1656e-03, 2.5990e-01, 6.2383e-01],
                        device=selected_pred.device,
                        dtype=selected_pred.dtype,
                        requires_grad=False)

                # level_loss = ((gt['level'][0,...]-pred['level']).abs()/std_tensor).pow(2).mean()
                # level_loss = ((gt['level'][0,...]-pred['level']).abs()/std_tensor).pow(2).mean()
                level_loss = ((gt['level'][0,...]-pred['level']).abs()/std_tensor).abs().pow(2).mean(axis=[1,2,3])
                level_loss = (level_loss /loss_means_level).mean()

                loss = torch.nanmean(torch.stack([loss, level_loss]))
            return loss
        else:
            std_tensor_surface = torch.tensor([3.35e4, 4.44, 3.83, 92.4], device=pred['level'].device,
                    dtype=pred['level'].dtype, requires_grad=False).unsqueeze(1).unsqueeze(2)
            # std of 
            # geopotential, specific_humidity, temperature, u_component_of_wind, v_component_of_wind 
            std_tensor_level = torch.tensor([2.625e+04, 0.002213,36.25, 7.481, 4.745], device=pred['level'].device,
                    dtype=pred['level'].dtype, requires_grad=False).unsqueeze(1).unsqueeze(2).unsqueeze(3)

            loss_means_surface = torch.tensor([3.3740e-04, 6.2160e-01, 7.8206e-01, 1.5008e-03], device=pred['level'].device,
                    dtype=pred['level'].dtype, requires_grad=False)
            loss_means_level = torch.tensor([2.1723e-04, 8.4831e-02, 2.1656e-03, 2.5990e-01, 6.2383e-01], device=pred['level'].device,
                    dtype=pred['level'].dtype, requires_grad=False)
            loss_means_surface = torch.tensor([1.0, 1.0, 1.0, 1.0], device=pred['level'].device,
                    dtype=pred['level'].dtype, requires_grad=False)
            loss_means_level = torch.tensor([1.0,1.0,1.0,1.0,1.0], device=pred['level'].device,
                    dtype=pred['level'].dtype, requires_grad=False)

            #loss_surface = ((pred['surface']-gt['surface'].squeeze())/std_tensor_surface).pow(2)
            loss_surface = ((pred['surface']-gt['surface'].squeeze())/std_tensor_surface).abs().pow(2).mean(axis=[1,2])/loss_means_surface

            #loss_level = ((pred['level']-gt['level']) / std_tensor_level).pow(2)
            loss_level = ((pred['level']-gt['level']) / std_tensor_level).abs().pow(2).mean(axis=[0,2,3,4])/loss_means_level
            loss = torch.nanmean(torch.cat([loss_surface.flatten(), loss_level.flatten()]))
            # weighted_error = (pred- gt).abs().pow(2)#.mul(self.loss_coeffs.to(gt.dtype))
            # loss = sum(weighted_error.mean().values())
            return loss


    def compute_lat_weights_weatherbench(self, latitude_resolution: int) -> torch.tensor: 
        '''Calculate the area overlap as a function of latitude.
        The weatherbench version gives slightly different coeffs.
        '''
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

    def training_step(self, pred, batch, prior_known_dim=None, complement=False, known_region=False, known_atmosphere=False):
        # calling this function
        loss = self.loss(pred, batch["next_state"], timestamp=batch['timestamp'],
                complement=complement, prior_known_dim=prior_known_dim,
                known_region=known_region, known_atmosphere=known_atmosphere)
        return loss

    # def validation_step(self, pred, batch, denormalize, prior_known_dim=None,complement=True, known_region=False):
    #     loss = self.loss(pred, batch["next_state"], timestamp=batch['timestamp'], complement=complement,  prior_known_dim=prior_known_dim, known_region=known_region)
    #     return loss


if __name__ == '__main__':
    # import inspect
    # functions_list = inspect.getmembers(inspect.getmodule(inspect.currentframe()), inspect.isfunction)
    # for func_name, func in functions_list:
    #     print(func_name)
    print("=== Environment Info ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    import pkg_resources
    onnx2torch_version = pkg_resources.get_distribution("onnx2torch").version
    print(f"onnx2torch: {onnx2torch_version}")
    main()
