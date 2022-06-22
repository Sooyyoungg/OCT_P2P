import random
import torch
import pandas as pd
import tensorboardX
import cv2
from sklearn.metrics import mean_squared_error
import torchsummary
from torchvision import transforms
import gc

from Config import Config
from DataSplit import DataSplit
from model import Pix2Pix
import networks

def main():
    config = Config()
    device = torch.device('cuda:{}'.format(config.gpu_ids[0])) if config.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
    print(device)

    ## Data Loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
    train_data = DataSplit(data_dir=config.train_root, transform=transform)
    valid_data = DataSplit(data_dir=config.valid_root, transform=transform)

    data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=16, pin_memory=False)
    data_loader_valid = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, shuffle=True, num_workers=16, pin_memory=False)
    print("Train: ", len(data_loader_train), "x", config.batch_size,"(batch size) =", len(train_data))
    print("Valid: ", len(data_loader_valid), "x", config.batch_size,"(batch size) =", len(valid_data))

    test_data = DataSplit(data_dir=config.test_root, transform=transform)
    data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=16,
                                                   pin_memory=False)
    print("Test: ", len(data_loader_test), "x", 1, "(batch size) =", len(test_data))

    ## Start Training
    model = Pix2Pix(config)
    model.to(device)

    # torchsummary.summary(model, (1, 66, 45), device='cpu')

    train_writer = tensorboardX.SummaryWriter(config.log_dir)

    print("Start Training!!")
    itr_per_epoch = len(data_loader_train)
    tot_itr = 0
    for epoch in range(config.n_epoch):
        for i, data in enumerate(data_loader_train):
            tot_itr += 1
            train_dict = model.train(i, data)

            fake_label = train_dict['fake_label']
            f_image = fake_label.detach().cpu().numpy()
            f_image = ((f_image + 1) / 2) * 255.0

            # save & print loss values
            train_writer.add_image('label', f_image, tot_itr, dataformats='NCHW')
            train_writer.add_scalar('Loss_G', train_dict['G_loss'], tot_itr)
            train_writer.add_scalar('Loss_D', train_dict['D_loss'], tot_itr)
            print("Epoch: %d/%d | itr: %d/%d | tot_itrs: %d | Loss_G: %.5f | Loss_D: %.5f"%(epoch+1, config.n_epoch, i+1, itr_per_epoch, tot_itr, train_dict['G_loss'], train_dict['D_loss']))

        with torch.cuda.device(device):
            gc.collect()
            torch.cuda.empty_cache()

        with torch.no_grad():
            valid_G_loss = 0
            valid_D_loss = 0
            for v, v_data in enumerate(data_loader_valid):
                val_dict = model.val(v_data)
                valid_G_loss += val_dict['G_loss']
                valid_D_loss += val_dict['D_loss']
                v_fake_label = train_dict['fake_label']

                v_f_image = v_fake_label.detach().cpu().numpy()
                v_f_image = ((v_f_image + 1) / 2) * 255.0

                train_writer.add_image('V_label', v_f_image, tot_itr, dataformats='NCHW')
                train_writer.add_scalar('V_Loss_G', train_dict['G_loss'], epoch)
                train_writer.add_scalar('V_Loss_D', train_dict['D_loss'], epoch)

            v_G_avg_loss = float(valid_G_loss / (v+1))
            v_D_avg_loss = float(valid_D_loss / (v+1))

            print("===> Validation <=== Epoch: %d/%d | Loss_G: %.5f | Loss_D: %.5f"%(epoch+1, config.n_epoch, v_G_avg_loss, v_D_avg_loss))

        networks.update_learning_rate(model.G_scheduler, model.optimizer_G)
        networks.update_learning_rate(model.D_scheduler, model.optimizer_D)

        # save model for each 10 epochs
        # if epoch % 10 == 0 or epoch == config.n_epoch - 1:
        #     torch.save(model.state_dict(), config.log_dir+'/{}_epoch{}_itr{}.pt'.format(config.data_name, epoch+1, tot_itr))
        #     with open(config.log_dir+'/latest_log.txt', 'w') as f:
        #         f.writelines('%d, %d'%(epoch+1, tot_itr))

        # save best performance model
        # if valid_rmse < min_v_rmse:
        #     min_v_rmse = valid_rmse
        #     torch.save(model.state_dict(), config.log_dir+'/{}_best_epoch{}_itr{}_rmse{}.pt'.format(config.data_name, epoch+1, tot_itr, min_v_rmse)) # SEM_best_epoch80_itr123124.pt


    ## Test
    with torch.no_grad():
        print("Testing")
        for i, data in enumerate(data_loader_test):
            test_dict = model.test(data)
            sub = data['sub']

            fake_label = test_dict['fake_label']

            # post-processing
            fake_label = fake_label.detach().cpu().numpy()
            fake_label = ((fake_label + 1) / 2) * 255.0

            # image 저장
            print(i, "th image save")
            cv2.imwrite('{}/{}'.format(config.test_img_dir,sub), fake_label)

if __name__ == '__main__':
    main()