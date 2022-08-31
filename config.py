class Config:

    gamma = 0.5 # balances complexity and reconstruction loss! higher gamma => more log likelihood loss!

    
    # TODO maybe make the log class a string depending on the inputs of the config file!
    # log_path = './weights_sslab12_20mio_gamma_0999'
    # log_path = './weights_triplets_50mio_run2'
    log_path = './weights_test'
    # log_path = "./weights_things_behavior_8196bs_adaptive_halfbs"

    modality = "behavior" 

    model_name = 'model_epoch_9.tar'
    load_model = False

    tensorboard = True
    

    init_dim = 100
    prior = 'sslab'

    stability_time = 2000

    feature_path = '../THINGS/vgg_bn_features12/features.npy'
    # feature_path = '/LOCAL/fmahner/THINGS/vgg_bn_features6/features.npy'

    # triplet_path = "../THINGS/triplets12_bn_adaptive_50mio/"
    triplet_path = "../THINGS/things_behavior_triplets"

    val_split = .1
    rnd_seed= 42
    shuffle_dataset = True

    batch_size = 8196
    # batch_size = 256
    n_epochs = 5000
    n_samples = 50_000_000


    params_interval = 200
    checkpoint_interval = 1000 # we dont save the model in principle!
    fresh=True

