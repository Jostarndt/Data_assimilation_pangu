#import era5
from era5 import Era5Forecast
from typing import Callable, Dict, List
from pathlib import Path
import re
from datetime import datetime
import torch
import xarray as xr
import os


import pdb


class ERA5IMERGDataset:
    def __init__(
        self, 
        imerg_data_dir: str = "/srv/data/IMERG/",
        domain: str = "train",
        filename_filter: Callable | None = None,
        lead_time_hours: int = 24,
        variables: Dict[str, List[str]] | None = None,
        **era5_kwargs  # Pass through all other ERA5 arguments
    ):
        self.imerg_data_dir = Path(imerg_data_dir)
        
        if domain == "train":
            split_min = 1998
            split_max = 2017
        elif domain ==  "val":# This works fine
            split_min = 2019#2018#2019
            split_max = 2019#only 1 year 
        elif domain == "test":
            split_min = 2020#2019#delivers 2020
            split_max = 2020
        else:
            raise ValueError(domain+" is not a known domain")
        filter_function = lambda x: split_min <= int(x.split('_')[2]) <= split_max
        #filter_function= lambda x: not ("2019" in x or "2020" in x or "2021" in x)
        self.era5_dataset = Era5Forecast(
                domain=domain,
                lead_time_hours=lead_time_hours,
                variables=variables,
                filename_filter=filter_function,
                norm_scheme=None,
                **era5_kwargs
                )
        self.era5_dataset = StridedDatasetWrapper(self.era5_dataset, stride=4)
        # Get available IMERG timestamps
        self.available_imerg_timestamps = self._get_available_imerg_timestamps()
        ait = sorted(self.available_imerg_timestamps)

        ait = [ts for ts in ait if split_min <= datetime.fromtimestamp(ts).year <= split_max]
        #TODO 
        self.available_imerg_triplets = [(ait[i], ait[i+1], ait[i+2]) for i in range(len(ait)-2)]
        #pdb.set_trace()
        '''
        self.__getitem__(0)
        self.__getitem__(1)
        '''
        
        # Build mapping of valid indices (only samples with IMERG data)
        
        print(f"ERA5 dataset has {len(self.era5_dataset)} samples")
        print(f"Found {len(self.available_imerg_timestamps)} IMERG files")
        pdb.set_trace()
        norm_file_path = os.path.dirname(__file__) + "incl_precip.pt"
        pangu_stats = torch.load(norm_file_path, weights_only=True)
        self.norm_scheme = "pangu"
        self.data_mean = TensorDict(
                surface=pangu_stats["surface_mean"],
                level=pangu_stats["level_mean"]
                )
        self.data_std = TensorDict(
                surface=pangu_stats["surface_std"],
                level=pangu_stats["level_std"],
                )

    def normalize(self, batch):
        if self.norm_scheme is None:
            return batch
        device = list(batch.values())[0].device

        state_means = self.data_mean.exclude("constant_surface").to(device)
        state_stds = self.data_std.exclude("constant_surface").to(device)


        if "surface" in batch:
            # we can normalize directly
            return (batch - means) / stds
        #print(batch)
        #pdb.set_trace()
        #print(means)
        #print(batch.keys())
        #out = {k: ((v - means) / stds if "state" in k else v) for k, v in batch.items()}
        out = {}
        for k, v in batch.items():
            if "state" in k:
                out[k] = (v - state_means) / state_stds
            else:
                out[k] = v
        return out


 
    def denormalize(self, batch):
        device = list(batch.values())[0].device
        means = self.data_mean.to(device)
        stds = self.data_std.to(device)
        if "surface" in batch:
            # we can denormalize directly
            return batch * stds + means

        out = {k: (v * stds + means if "state" in k else v) for k, v in batch.items()}
        return out


    def _get_available_imerg_timestamps(self):
        """Extract available timestamps from IMERG filenames"""
        imerg_files = list(self.imerg_data_dir.glob("3B-DAY*.nc4"))
        
        available_timestamps = set()
        date_pattern = r'3B-DAY\.MS\.MRG\.3IMERG\.(\d{8})-S\d{6}-E\d{6}\.V\d{2}B\.nc4'
        
        for file in imerg_files:
            match = re.search(date_pattern, file.name)
            if match:
                date_str = match.group(1)  # YYYYMMDD
                dt = datetime.strptime(date_str, "%Y%m%d")
                timestamp = int(dt.timestamp())
                available_timestamps.add(timestamp)
        
        return available_timestamps
    
   
    def _timestamp_to_imerg_file(self, timestamp):
        """Convert Unix timestamp to IMERG filename"""
        if isinstance(timestamp, torch.Tensor):
            timestamp = timestamp.item()
        
        dt = datetime.fromtimestamp(timestamp)
        date_str = dt.strftime("%Y%m%d")
        filename = f"3B-DAY.MS.MRG.3IMERG.{date_str}-S000000-E235959.V07B.nc4"
        return self.imerg_data_dir / filename
    
    def _load_imerg_data(self, timestamp):
        """Load and process IMERG data for given timestamp"""
        imerg_file = self._timestamp_to_imerg_file(timestamp)
        
        if not imerg_file.exists():
            raise FileNotFoundError(f"IMERG file not found: {imerg_file.name}")
        print(imerg_file)
        ds = xr.open_dataset(imerg_file)
        #print(ds.head())
        ds = ds.sortby('lat', ascending=False)#i dont get why, but plot looks better
        ds = ds.assign_coords(lon=((ds.lon % 360)))
        ds.sortby('lon')

        
        # Extract precipitation: shape (1, 3600, 1800)
        precip_data = ds['precipitation'].values
        
        # Reshape: (time, lon, lat) -> (channels=1, levels=1, lat, lon)
        precip_reshaped = precip_data[0].T  # (3600, 1800) -> (1800, 3600)
        precip_tensor = torch.from_numpy(precip_reshaped).unsqueeze(0).unsqueeze(0)  # (1, 1, 1800, 3600)
        
        ds.close()
        return precip_tensor.to(torch.bfloat16)
    
    def __len__(self):
        return len(self.available_imerg_triplets)-1#Because era5 gets sampled with index+1
    
    def __getitem__(self, idx):
        """Get combined ERA5+IMERG batch"""
        # Map to valid ERA5 index
        #print("INDEX: ", idx)
        era5_sample = self.era5_dataset[idx+1]#I dont get why, but plot looks better like this
        print(datetime.fromtimestamp(era5_sample["timestamp"]))
        print(datetime.fromtimestamp(self.available_imerg_triplets[idx][1]))
        # assert (abs(era5_sample["timestamp"] - self.available_imerg_triplets[idx][1] - 86400) <= 7200), "Mismatch between era5 and imerg batches"
        
        imerg_files = [self._load_imerg_data(file_path) for file_path in self.available_imerg_triplets[idx]]

        # Move to same device as ERA5 data
        device = era5_sample['state']['surface'].device

        imerg_files = [imerg_file.to(device) for imerg_file in imerg_files]
        
        # Add IMERG to each state TensorDict
        for i, state_key in enumerate(['prev_state', 'state', 'next_state']):
            if state_key in era5_sample:
                era5_sample[state_key]['imerg'] = imerg_files[i]
        
        return era5_sample




class StridedDatasetWrapper:
    def __init__(self, dataset, stride=1):
        self.dataset = dataset
        self.stride = stride
        self._length = len(dataset) // stride
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        return self.dataset[idx * self.stride]
    
    def __getattr__(self, name):
        # Delegate all other attributes/methods to original dataset
        return getattr(self.dataset, name)
