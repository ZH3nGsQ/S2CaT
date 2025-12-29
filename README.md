# S2CaT: Spatial-Spectral-based CNN and Transformer Network for Hyperspectral Image Classification


Here is a pytorch implementation of S2CaT.

## requirements
Run the following command to install required dependencies：
``` bash
cd S2CaT
pip install -r requirements.txt
```

## Training and Testing

If you want to train this model, please execute the script below:
```bash
sh run.sh
```
or:
```bash
python S2CAT.py
```

## Datasets and Weights
please put your own HSI dataset to data/ and modify the train_test.py to train the model. The weight of model will be saved at weights/.
