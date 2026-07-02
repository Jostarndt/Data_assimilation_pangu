# Comparing 3D attention on different datasets.

## Structure of this project


my_ml_project/
│
├── data/
│   ├── source1/               
│   │   ├── raw/               
│   │   ├── processed/         
│   │   └── README.md          
│   └── source2/               
│       ├── raw/
│       ├── processed/
│       └── README.md
│
├── models/                    
│   ├── model_a.py             
│   ├── model_b.py             
│   └── README.md              
│
├── notebooks/                 
│   ├── exploratory_analysis.ipynb
│   └── comparison.ipynb
│
├── scripts/                   
│   ├── training.py            # Main training script
│   ├── slurm_job.sh           # SLURM job submission script
│   └── slurm_array_job.sh     # SLURM array job submission script
│
├── apptainer/                 
│   ├── my_container.def       # Apptainer definition file
│   └── README.md              # Instructions for building the container
│
├── requirements.txt           
├── config.yaml                
└── README.md


## Package installation
look at https://docs.astral.sh/uv/getting-started/features/#the-pip-interface

install uv

uv venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio
uv pip install torch_geometric
uv pip install PyYAML
uv pip install wandb


uv pip freeze > requirements.txt
