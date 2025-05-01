# Experimenting with Cliqueformer

This repository is a fork of the official implementation of Cliqueformer, as described in the paper: **Cliqueformer: Model-Based Optimization with Structured Transformers**  by Jakub Grudzien Kuba, Pieter Abbeel, Sergey Levine  

## Installation

```bash
git clone https://github.com/znowu/cliqueformer-code.git
cd cliqueformer-code
pip install -r requirements.txt
```

## Downloading assets for DNA Enhancers experiments

```bash
cd scrape/Bioseq
python download_data.py
python download_model.py
cd ../..
```

## Train or optimize an MBO model

```bash
python training.py
```
```bash
python optimize.py
```

You can change the task you want to solve by changing the config file in *training.py* and *optimize.py*. For example, for Superconductor: *'configs/superconductor/cliqueformer.py'*.
