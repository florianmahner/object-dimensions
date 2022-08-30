from config import Config
import numpy as np
from main import train

def main():

    # have to be careful with the beta

    n_dimensions = np.arange(10,100,1)

    for dim in n_dimensions:

        cfg = Config()
        cfg.gamma = 0.3
    
        cfg.fresh = True
        cfg.prior = 'exp'
        cfg.feature_path = '/LOCAL/fmahner/THINGS/vgg_bn_features12/features.npy'
        cfg.n_samples = 1_000_000
        cfg.stability_time = 500
        cfg.rnd_seed = 42
        cfg.batch_size = 8096 * 4
        cfg.log_path = './weights_exp_12_1mio_ndim' + str(dim)
        cfg.tensorboard = False


        cfg.init_dim = dim

        

        train(cfg)




if __name__ == '__main__':
    main()