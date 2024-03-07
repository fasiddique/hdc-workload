# %%
import torchhd
import torchvision

DATASETS = ["MNIST", "ISOLET", "EMG_Hand", "UCIHAR"]


def load_dataset(name):
    """
    Load dataset by name
    """
    if name in DATASETS:
        if name == "MNIST":
            train = torchvision.datasets.MNIST(
                root="./dataset", train=True, download=True
            )
            test = torchvision.datasets.MNIST(
                root="./dataset", train=False, download=True
            )
        elif name == "ISOLET":
            train = torchhd.datasets.ISOLET(root="./dataset", train=True, download=True)
            test = torchhd.datasets.ISOLET(root="./dataset", train=False, download=True)
        elif name == "EMG_Hand":
            data = torchhd.datasets.EMGHandGestures(root="./dataset", download=True)
            n_split = int(len(data) * 0.75)
            train, test = data[:n_split], data[n_split:]
        elif name == "UCIHAR":
            train = torchhd.datasets.UCIHAR(root="./dataset", train=True, download=True)
            test = torchhd.datasets.UCIHAR(root="./dataset", train=False, download=True)
    else:
        raise NotImplementedError(f"Dataset {name} not implemented!")

    if name in ["EMG_Hand"]:
        data_train, data_test = (train[0].flatten(1).int(), train[1]), (
            test[0].flatten(1).int(),
            test[1],
        )
    else:
        data_train, data_test = (train.data.flatten(1), train.targets), (
            test.data.flatten(1),
            test.targets,
        )
    return data_train, data_test


def save_dataset(data_train, data_test, name):
    filename = '../CPP/dataset/' + name + '/train.val'
    with open(filename, 'w') as file:
        for sample in data_train[0]:
            line = ""
            for value in sample: 
                line += str(value.item()) + " "
            file.write(line + "\n")

    filename = '../CPP/dataset/' + name + '/train.label'
    with open(filename, 'w') as file:
        for label in data_train[1]:
            line = str(label.item())
            file.write(line + "\n")

    filename = '../CPP/dataset/' + name + '/test.val'
    with open(filename, 'w') as file:
        for sample in data_test[0]:
            line = ""
            for value in sample: 
                line += str(value.item()) + " "
            file.write(line + "\n")

    filename = '../CPP/dataset/' + name + '/test.label'
    with open(filename, 'w') as file:
        for label in data_test[1]:
            line = str(label.item())
            file.write(line + "\n")

    test_size = data_test[1].size()[0]
    train_size = data_train[1].size()[0]
    sample_size = data_train[0][1].size()[0]
    line = str(test_size) + "\n" + str(train_size) + "\n" + str(sample_size)
    filename = '../CPP/dataset/' + name + '/parameters'
    with open(filename, 'w') as file:
        file.write(line + "\n")
