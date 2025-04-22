# CS7150 Project: Transformer based Time Series Latent Diffusion Model for Bio-signal generation

 
- Models are separated by their individual folders of the same name
- `trainer1D.ipynb` and `trainer2D.ipynb` (where applicable) are the main runner files showcasing the training and final output of each model
- Samples of generated signals are stored with class labels in each model directory
- `Metrics` folder contains code for evaluating the similarity metrics. It reads the stored signals from each model directory (only 1D training).

## Note
All the data could not be uploaded due to size constraints. Only S class signals are uploaded so if running the experiments, indicate you want to run the models on class S in the runner files.