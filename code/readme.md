# SAF-DETR

Implementation of CSSFAN . 

CSSFAN is implemented on top of MMDetection v3.3.0. The essential source codes have been released, and the model can be reproduced by following the official MMDetection tutorial guidelines, where the provided codes only need to be placed into the corresponding directories and registered accordingly.

## Training and Validation

- **Training (using NEU-DET dataset as an example):**
  
  ```bash
  python train.py configs/saf_detr/regnet_haem_safenc_sda_100e_NEU-DET.py
  ```

- **Validation (using NEU-DET dataset as an example):**
  
  ```bash
  python test.py configs/saf_detr/regnet_haem_safenc_sda_100e_NEU-DET.py
  ```
