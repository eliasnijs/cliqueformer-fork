import wandb

run = wandb.init(project="Diffusion-DNA-RNA")
artifact = run.use_artifact('fderc_diffusion/Diffusion-DNA-RNA/DNA-model:v0')
dir = artifact.download()
wandb.finish()

