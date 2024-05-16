# %%
import math
import torch
import tqdm
import model
import utils
import numpy as np

from debug_utils import *

def encode_test():
    # Initialize inputs for testing 
    n_data = 1 # 1735
    n_class =  torch.tensor(5)
    n_lv =  21
    n_id =  1024
    N_DIM =  2048
    BINARY =  False

    # Define the shape
    shape = torch.Size([n_data, n_id])

    # Define the fixed value
    fixed_value = 0

    # Create a tensor with all values set to the fixed value and the specified shape
    fixed_value_tensor = torch.full(shape, fixed_value)

    # HDC Model
    hdc_model = model.HDC_ID_LV(
        n_class=n_class, n_lv=n_lv, n_id=n_id, n_dim=N_DIM, binary=BINARY
    )

    # HDC Encoding Step
    fixed_value_tensor_enc = hdc_model.encode(fixed_value_tensor)

    print("[DEBUG] fixed_value_tensor_enc.shape =", fixed_value_tensor_enc.shape)

    print("[DEBUG] aggregate_tensor_mod(fixed_value_tensor_enc) =", aggregate_tensor_mod(fixed_value_tensor_enc))

    # Assuming fixed_value_tensor_enc is your tensor
    fixed_value_tensor_enc_np = fixed_value_tensor_enc.cpu().numpy()

    print(fixed_value_tensor_enc_np)

    # Save to a file
    with open('../CPP/tensor_data.bin', 'wb') as f:
        f.write(fixed_value_tensor_enc_np.tobytes())

if __name__ == "__main__": 
    encode_test()