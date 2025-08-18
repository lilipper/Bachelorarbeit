import pytorch_lightning as pl
from torch.utils.data import DataLoader
from Bachelorarbeit.Adapter.ControlNet.load_thz_dataset_one_target import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, \
    EulerDiscreteScheduler
import torch


# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", scheduler=scheduler, torch_dtype=torch.float16)
pipe.enable_xformers_memory_efficient_attention()
vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet
# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
