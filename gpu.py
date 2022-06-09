
import torch

if __name__ == "__main__":
    print(torch.cuda.is_available())

    print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    print(torch.cuda.get_device_name(0))

    print(torch.version.cuda)  # causes error

    print(torch.cuda.get_arch_list())

    print(torch.cuda.device_count())