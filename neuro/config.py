import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_grad_norm = 0.1

import os
n_workers = os.cpu_count()

method_name = ['f2']
method_args = [f'--threads {n_workers} --time-limit 1']
