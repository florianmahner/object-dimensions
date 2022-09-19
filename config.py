class Config:


    # log_path = './learned_embeddings/weights_vgg_12_8196bs_adaptive'
    log_path = './learned_embeddings/weights_test'


    modality = "deep" 
    load_model = True
    fresh = False

    tensorboard = True
    

    init_dim = 100
    prior = 'sslab'

    stability_time = 2000

    feature_path = '../THINGS/vgg_bn_features_12/features.npy'
    triplet_path = '../THINGS/triplets_12_20mio/'

    val_split = .1
    rnd_seed= 42
    shuffle_dataset = True

    batch_size = 4096
    n_epochs = 1

    params_interval = 1
    checkpoint_interval = 1  # we dont save the model in principle!
    

