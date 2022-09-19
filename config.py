class Config:


    # log_path = './learned_embeddings/weights_vgg_12_8196bs_adaptive'
    log_path = './learned_embeddings/weights_test'


    modality = "deep" 
    model_name = 'model_epoch_9.tar'
    load_model = False

    tensorboard = True
    

    init_dim = 100
    prior = 'sslab'

    stability_time = 2000


    triplet_path = "../THINGS/triplets_12_20mio/"

    val_split = .1
    rnd_seed= 42
    shuffle_dataset = True

    batch_size = 8196
    n_epochs = 5

    params_interval = 200
    checkpoint_interval = 1000 # we dont save the model in principle!
    fresh=True

