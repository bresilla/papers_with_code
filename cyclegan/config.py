#hyperparameters

learning_rate = 2e-4
lambda_identity = 0.0
lambda_cycle = 10.0
labda_adversarial = 1.0
lambda_pixel = 100.0

batch_size = 4
num_workers = 8
num_epochs = 100

device = "cuda"

#datapath
dataset = "apple2orange"
# data_root = f"/doc/CODE/LEARN/pytorch/cyclegan/data/cyclegan/apple2orange/apple2orange"
data_root = f"/doc/code/train/pytorch/cyclegan/data/cyclegan/{dataset}/{dataset}"