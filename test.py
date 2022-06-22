import time

import cv2
import torch
import pandas as pd
import numpy as np
from torchvision.transforms import transforms
from torchvision.utils import save_image

from Config import Config
from DataSplit import DataSplit
from model import Pix2Pix


def main():
    config = Config()
    device = torch.device('cuda:{}'.format(config.gpu_ids[0])) if config.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
    print(device)

    ## Data Loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
    test_data = DataSplit(data_dir=config.test_root, transform=transform)
    data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=16, pin_memory=False)
    print("Test: ", len(data_loader_test), "x", 1,"(batch size) =", len(test_data))

    ## Start Training
    model = Pix2Pix(config)
    model.load_state_dict(torch.load(config.log_dir+'/SEM_best_epoch72_itr14040000_rmse2.812597536459417.pt'))
    model.to(device)

    print("Start Testing!!")
    tot_itr = 0
    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(data_loader_test):
            tot_itr += i
            test_dict = model.test(data)

            fake_depth = test_dict['fake_depth']
            sub = test_dict['sub'][0]

            # post-processing
            fake_depth = fake_depth[0, 0, :, :].detach().cpu().numpy()
            fake_depth = ((fake_depth + 1) / 2) * 255.0

            # image 저장
            print(i, "th image save")
            cv2.imwrite('{}/{}'.format(config.test_img_dir,sub), fake_depth)

    end_time = time.time()
    print("Testing Time: ", end_time - start_time)

if __name__ == '__main__':
    main()