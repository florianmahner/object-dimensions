class Config:


    # log_path = './learned_embeddings/weights_vgg_12_8196bs_adaptive'
    # log_path = './learned_embeddings/weights_vgg_12_32384bs_adaptivehalf'
    # log_path = './learned_embeddings/weights_vgg_12_512bs'
    # log_path = './learned_embeddings/weights_adaptive_bs'

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

    rnd_seed= 42

    batch_size = 256
    n_epochs = 3000

    params_interval = 50
    checkpoint_interval = 50  # we dont save the model in principle!
    

