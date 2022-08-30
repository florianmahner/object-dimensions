from config import Config
import numpy as np
from main import train

def main():

    # gamma_array = np.linspace(0.95, 0.999, 10)
    gamma_array = []

    for gamma in gamma_array:
    
        print('Take new Gamma {} and train the model'.format(gamma))

        cfg = Config()
        cfg.gamma = gamma
        cfg.fresh = True
        cfg.prior = 'exp'
        cfg.feature_path = '/LOCAL/fmahner/THINGS/vgg_bn_features12/features.npy'
        cfg.n_samples = 1_000_000
        cfg.stability_time = 500
        cfg.rnd_seed = 42
        cfg.batch_size = 8096 * 4
        gamma_str = str(gamma)[:5].replace('.', '')
        cfg.log_path = './weights_exp_12_1mio_gamma' + gamma_str
        
        train(cfg)


if __name__ == '__main__':
    main()