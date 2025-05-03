# FedHuman: 3D Reconstruction with Federated Learning

  <p align="center">
    <a href="https://adamoorsini.com/"><strong>Adamo Orsini</strong></a>
    ·    
    <a href="https://sadmanpinon.github.io/Homepage/"><strong>Sadman Pinon</strong></a>
    ·
    <strong>Terrance Liang</strong>
    ·
  </p> 

  <p align="center">
    <a href="https://youtu.be/UZByQ7lvVAY"><strong>Paper Presentation Video</strong></a>
  </p> 

## Important files
- fed_entrypoint.py: Main methods for the FL environment 
- lib/flower: Folder containing client, server, and strategies used by the FL environment 

## Usage

1. Download the THuman2.0 dataset preprocessed for GHG using the steps shown [here](https://github.com/humansensinglab/Generalizable-Human-Gaussians/blob/main/INSTALL.md#use-the-pre-processed-thuman-20-dataset).
2. Create a conda environment or python venv,  then run using:

```
pip install -e .
flwr run
```

