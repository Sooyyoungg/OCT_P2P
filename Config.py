class Config:
    ## dataset parameters
    train_root = '/scratch/connectome/conmaster/Pycharm_projects/CT_Joonwoo/data/train'
    valid_root = '/scratch/connectome/conmaster/Pycharm_projects/CT_Joonwoo/data/val'
    test_root = '/scratch/connectome/conmaster/Pycharm_projects/CT_Joonwoo/data/test'

    # output directory
    log_dir = '/scratch/connectome/conmaster/Pycharm_projects/CT_Joonwoo/log'
    # img_dir = '/scratch/connectome/conmaster/Pycharm_projects/CT_Joonwoo/Generated_images'
    # valid_img_dir = '/scratch/connectome/conmaster/Pycharm_projects/CT_Joonwoo/Best_Train_images'
    test_img_dir = '/scratch/connectome/conmaster/Pycharm_projects/CT_Joonwoo/Tested_images'

    ## basic parameters
    gpu_ids = [7]
    n_epoch = 1
    n_iter = 100
    n_iter_decay = 100
    batch_size = 1
    lr = 1e-3
    lr_policy = 'step'
    lr_decay_iters = 50
    beta1 = 0.5
    pool_size = 50
    image_display_iter = 100
    gan_mode = 'vanilla'
    lambda_L1 = 100

    # model parameters
    input_nc = 1
    output_nc = 1
    ngf = 64
    ndf = 64
    netG = 'unet_256'
    netD = 'basic'
    n_layers_D = 3
    initial = True     # Initialize the Generator
    norm = 'instance'   # [instance | batch | none]
    init_type = 'normal' # [normal | xavier | kaiming | orthogonal]
    init_gain = 0.02    # scaling factor for normal, xavier and orthogonal
    no_dropout = 'store_true'   # no dropout for generator