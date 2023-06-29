#hyperparameters
batch_size = 8
learning_rate = 2e-4
num_workers = 2
l1_lambda = 100
lambda_gp = 10
image_size = [256, 256]
in_channels = 3
num_epochs = 100
features_g = 64
features_d = 64
train_test_split = 0.9
dataset = "maps" # facades, maps, edges2shoes, cityscapes
base_path = "/doc/code/train/pytorch/pix2pix"
data_dir = f"{base_path}/data/{dataset}/{dataset}"
save_images = f"{base_path}/data/saved_images"