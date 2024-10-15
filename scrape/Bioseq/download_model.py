import wandb

run = wandb.init(project="Diffusion-DNA-RNA")
artifact = run.use_artifact('fderc_diffusion/Diffusion-DNA-RNA/DNA-model:v0')
dir = artifact.download()
wandb.finish()


from grelu.lightning import LightningModel
model = LightningModel.load_from_checkpoint("scrape/Bioseq/artifacts/DNA-model:v0/reward_model.ckpt")