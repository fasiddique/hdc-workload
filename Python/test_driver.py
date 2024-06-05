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


def train_test(): 
    # Initialize inputs for testing 
    n_lv =  5 # 21
    N_DIM =  2048
    BINARY =  False
    
    ds_train, ds_test = utils.load_dataset(name="EMG_Hand")

    n_class = ds_train[1].max() + 1 # Number of classes in the dataset
    n_id = ds_train[0].shape[1] # Number of input dimensions 
    
    if ds_train[0].dtype in [torch.int, torch.uint8, torch.int64]:
        n_lv = int(ds_train[0].max()) + 1 
 
    print("[DEBUG] n_class = ", n_class)
    print("[DEBUG] n_lv = ", n_lv)
    print("[DEBUG] n_id = ", n_id)
    print("[DEBUG] N_DIM = ", N_DIM)
    print("[DEBUG] BINARY = ", BINARY)
    print("[DEBUG] ds_train[0].shape = ", ds_train[0].shape)
    print("[DEBUG] ds_train[1].shape = ", ds_train[1].shape)
    print("[DEBUG] ds_test[0].shape = ", ds_test[0].shape)
    print("[DEBUG] ds_test[1].shape = ", ds_test[1].shape)

    # HDC Model
    hdc_model = model.HDC_ID_LV(
        n_class=n_class, n_lv=n_lv, n_id=n_id, n_dim=N_DIM, binary=BINARY
    )

    # HDC Encoding Step
    train_enc = hdc_model.encode(ds_train[0])
    test_enc = hdc_model.encode(ds_test[0])

    # Init. Training
    model.train_init(hdc_model, inp_enc=train_enc, target=ds_train[1])

    test_acc = model.test(hdc_model, inp_enc=test_enc, target=ds_test[1])
    print(f"Init. test acc. is {test_acc:.4f}")
        
    # Re-training
    train_epochs = 20
    val_epochs = 5


    for i in tqdm.tqdm(range(train_epochs)):
        model.train(hdc_model, inp_enc=train_enc, target=ds_train[1])

        if (i + 1) % val_epochs == 0:
            test_acc = model.test(hdc_model, inp_enc=test_enc, target=ds_test[1])
            print(f"Test acc. @ epoch {i+1}/{train_epochs} is {test_acc:.4f}")


    test_acc = model.test(hdc_model, inp_enc=test_enc, target=ds_test[1])
    print(f"Init. test acc. is {test_acc:.4f}")

    if BINARY:
        hdc_model.class_hvs = hdc_model.class_hvs.sign()


if __name__ == "__main__": 
    train_test()