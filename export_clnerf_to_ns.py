import torch
from train_ngpgv2_CLNerf import NeRFSystem      # your LightningModule
from nerfstudio.engine.pipelines.vanilla import VanillaPipeline

# 1a) Load your hyperparameters exactly as at training time
hparams = {...}  # for instance, dict(root_dir=..., dataset_name=..., …)

# 1b) Load the Lightning checkpoint
system = NeRFSystem.load_from_checkpoint("epoch=19.ckpt", **hparams)
model  = system.model.eval().to("cuda")

# 2) Wrap it in Nerfstudio’s VanillaPipeline
pipeline = VanillaPipeline(model=model, use_opengl_format=False)

# 3) Save only the `model` state dict under Nerfstudio’s expected key
ns_ckpt = {"model": pipeline.model.state_dict()}
torch.save(ns_ckpt, "nerfstudio_model.ckpt")
