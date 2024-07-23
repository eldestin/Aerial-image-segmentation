class Con:
    num_frames = 3
    num_ref_frames = 3
    deep_update_prob = 0.2
    max_jump = 5
    cropsize = 512
    save_top_k = 1
    monitor = "val_mIoU"
    monitor_mode = "max"
    save_last = True
    weights_name = "xmem"
    weights_path = "model_weights/uavid/{}".format(weights_name)
    start_warm = 2000
    end_warm = 7000
    num_classes = 8
    log_name = "uavid_xmem"
    max_epoch = 40
    check_val_every_n_epoch = 1
    gpus = 'auto'
    lr = 1e-5
    weight_decay = 0.05
    gamma = 0.1
    steps = 2000
    uavid_root = "C:/Users/Franklin/code/Aerial image segmentation/Dataset/uavid_v1.5_official_release"