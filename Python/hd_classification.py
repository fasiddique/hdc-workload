import math
import torch
import tqdm
import model
import utils
from debug_utils import *

def save_hdc_params(dataset_name, n_dim=2048, binary=False, train_epochs=20, n_lv=32):
     filename = '../CPP/dataset/' + dataset_name + '/hdc_parameters'
     with open(filename, 'w') as file:
         line = str(n_dim)
         file.write(line + "\n")

         line = str(int(binary))
         file.write(line + "\n")

         line = str(train_epochs)
         file.write(line + "\n")

         line = str(n_lv)
         file.write(line + "\n")

def train_and_evaluate_hdc_model(dataset_name, n_dim=2048, binary=False, train_epochs=20, val_epochs=5):
    # Load dataset
    ds_train, ds_test = utils.load_dataset(name=dataset_name)
    utils.save_dataset(ds_train, ds_test, name=dataset_name)

    n_class = ds_train[1].max() + 1 # Number of classes in the dataset
    n_id = ds_train[0].shape[1] # Number of input dimensions

    # Data Quantization
    if ds_train[0].dtype in [torch.int, torch.uint8, torch.int64]:
        n_lv = int(ds_train[0].max()) + 1 # n_lv is the number of levels or quantization levels used in the dataset.
    else:
        n_lv = 32
        ds_train = (
            model.min_max_quantize(ds_train[0], int(math.log2(n_lv) - 1)),
            ds_train[1],
        )
        ds_test = (model.min_max_quantize(ds_test[0], int(math.log2(n_lv) - 1)), ds_test[1])

    save_hdc_params(dataset_name, n_dim, binary, train_epochs, n_lv)
    # HDC Model
    hdc_model = model.HDC_ID_LV(
        n_class=n_class, n_lv=n_lv, n_id=n_id, n_dim=n_dim, binary=binary
    )

    # HDC Encoding Step
    train_enc = hdc_model.encode(ds_train[0])
    test_enc = hdc_model.encode(ds_test[0])

    # Init. Training
    model.train_init(hdc_model, inp_enc=train_enc, target=ds_train[1])
    test_acc = model.test(hdc_model, inp_enc=test_enc, target=ds_test[1])
    print(f"INFO: Init. test acc. for {dataset_name} is {test_acc:.4f}")

    # Re-training
    for i in tqdm.tqdm(range(train_epochs)):
        model.train(hdc_model, inp_enc=train_enc, target=ds_train[1])

        if (i + 1) % val_epochs == 0:
            test_acc = model.test(hdc_model, inp_enc=test_enc, target=ds_test[1])
            print(f"INFO: Test acc. for {dataset_name} @ epoch {i+1}/{train_epochs} is {test_acc:.4f}")

    test_acc = model.test(hdc_model, inp_enc=test_enc, target=ds_test[1])
    print(f"INFO: Final test acc. for {dataset_name} is {test_acc:.4f}")

    if binary:
        hdc_model.class_hvs = hdc_model.class_hvs.sign()

# List of datasets to iterate over
datasets = ["EMG_Hand", "MNIST", "UCIHAR", "ISOLET"]

for dataset in datasets:
    train_and_evaluate_hdc_model(dataset_name=dataset)

