import os
import sys
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(THIS_DIR)
sys.path.append(MODULE_DIR)
from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer

PNG_FOLDER = os.environ.get("PNG_FOLDER")
png_csv_file = "/radraid/pteng/DeepLesion/pngs.csv"
df = pd.read_csv(png_csv_file)
file_list = df.png_path.tolist()
print(type(file_list),len(file_list))

vae = VQGanVAE(
    dim = 256,
    codebook_size = 65536
)

# train on folder of images, as many images as possible

trainer = VQGanVAETrainer(
    vae = vae,
    image_size = 128,             # you may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it
    file_list = file_list, #folder = PNG_FOLDER,
    batch_size = 4,
    grad_accum_every = 8,
    num_train_steps = 50000
).cuda()

trainer.train()