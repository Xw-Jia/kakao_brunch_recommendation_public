from pathlib import Path

import torch

dataset_root = Path('datasets')
models_root = Path('models')
result_path = Path('recommend.txt')

save_memory = True

heavy_user_threshold = 50
heavy_item_threshold = 250

mf_factors = 8
mf_iterations = 10

batch_size = 1024
num_workers = 64
optimizer = torch.optim.SGD
learning_rate = 0.1
device = 'cuda'
device_idx = '0,1,2,3'
num_gpu = len(device_idx.split(',')) if device == 'cuda' else 0

reader_count = 322614
article_count = 672797

gbm_n_estimators = 10000

train_sample_size = 5000
val_sample_size = 1000
submit_user_set = 'test'
candidate_size = 100
