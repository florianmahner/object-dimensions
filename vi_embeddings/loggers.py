import torch
import os
import sys
import logging
from collections import defaultdict
from abc import ABC, abstractmethod
import numpy as np
import operator

# TODO Have to still check with the step counter when logging per epoch vs per batch!

class Logger(ABC):    

    @property
    @abstractmethod
    def log_path(self):
        ...

    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def _make_dir(self):
        filename, file_extension = os.path.splitext(self.log_path)
        # create directory if log path is not a file
        if not file_extension and not os.path.exists(filename):
            os.makedirs(self.log_path)
        # otherwise create an empty file
        elif file_extension:
            try:
                dir_path = os.path.dirname(self.log_path)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                open(self.log_path, 'a').close()
            except OSError:
                print('Failed creating the file at {}'.format(self.log_path))

    def __call__(self, *args, **kwargs):
        return self.log(*args, **kwargs)


class DefaultLogger(Logger):
    def __init__(self, log_path):
        self._log_path = log_path
        self._make_dir()
        self.callbacks = defaultdict(list)
        self.extensions = defaultdict(Logger)
        # incremental counter for the number of logging operations
        self.step_ctr = 0
    
    @property
    def log_path(self):
        return self._log_path

    def add_callback(self, logger_key, query):
        if logger_key in self.callbacks:
            self.callbacks[logger_key].append(query)
        else:
            self.callbacks[logger_key] = query

    def add_logger(self, logger_key, logger, update_interval=1, callbacks=None):
        " Each logger gets a key associated with it "
        assert isinstance(logger, Logger), 'Each Logger must be an instance of {}'.format(Logger)
        self.extensions[logger_key] = (logger, update_interval)
        self.callbacks[logger_key] = callbacks

    def log(self, *args, **kwargs):
        self.step_ctr += 1
        for name, (logger, update_interval) in self.extensions.items():        
            if self.step_ctr % update_interval != 0:
                continue

            # iterate over all callbacks for that logger
            call_keys = self.callbacks.get(name) 
            if call_keys:
                # filter dict with call keys
                log_dict = {k: kwargs[k] for k in call_keys if k in kwargs}
                logger.log(*args, step=self.step_ctr, **log_dict)

            else: 
                # some loggers dont have callbacks and just log e.g. model parameters
                logger.log(*args, step=self.step_ctr, **kwargs)


class FileLogger(Logger):
    def __init__(self, log_path):
        self._log_path = os.path.join(log_path, 'training.log')
        self._make_dir()
        self._init_loggers()

    @property
    def log_path(self):
        return self._log_path
                
    def _init_loggers(self):
        logging.basicConfig(level=logging.INFO)        
        # If previous logger exists, we first need to delete the old root logger
        fileh = logging.FileHandler(self.log_path, 'w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fileh.setFormatter(formatter)
        log = logging.getLogger()  # root logger
        stdlog = logging.StreamHandler(sys.stdout)
        log.removeHandler(stdlog)  # prevents multiple console prints
        for handler in log.handlers[:]:  # remove all old handlers
            log.removeHandler(handler)
            log.addHandler(stdlog) # also print to stout    
            log.addHandler(fileh)      # set the new handler

    def log(self, step=None, prepend=None, *args, **kwargs):
        logging.info("")
        prepend = prepend + ' - ' if prepend else ''
        for a in args:
            a = a.replace("_", " ").capitalize()
            logging.info(prepend + a)
        for k, v in kwargs.items():
            k = k.replace("_", " ").capitalize()
            logging.info(f'{prepend}{k}: {v}')
        

class ParameterLogger(Logger):
    def __init__(self, log_path, model, attributes, ext='.npz'):
        " attributes can either be attributes as strings or a function that transforms attributes for storing"
        self._log_path = os.path.join(log_path, 'params')
        self._make_dir()
        self.attributes = attributes
        self.model = model
        self.ext = ext
        self._check_attributes()
        self._check_ext()

    @property
    def log_path(self):
        return self._log_path

    def _check_ext(self):
        assert self.ext in ['.txt', '.npz'], 'Extension must be either .txt or .npz'

    def _check_attributes(self):
        if not isinstance(self.attributes, list):
            self.attributes = list(self.attributes)
        for v in self.attributes:
            assert isinstance(v, str)
            assert operator.attrgetter(v)(self.model) is not None, 'Model {} has no attribute of name {}'.format(self.model, v)

    def _save_params(self, param_dict, step):
        for key, val in param_dict.items():
            key = key + '_' + str(step)
            if self.ext == 'npz':
                np.savez(os.path.join(self.log_path, key + self.ext), val)
            else:
                np.savetxt(os.path.join(self.log_path, key + self.ext), val)

    def log(self, step=None, *args, **kwargs):
        param_dict = dict()
        for attr in self.attributes:            
            # param = getattr(self.model, attr)
            param = operator.attrgetter(attr)(self.model)

            if isinstance(param, torch.nn.Module):
                param = param.weight
            if isinstance(param, torch.Tensor):
                if param.requires_grad():
                    param = param.detach().cpu()
                param = param.numpy()    

            # if attribute is a function
            if callable(param):
                param_dict.update(param())
            else:
                param_dict[attr] = param

        self._save_params(param_dict, step)
    
class TensorboardLogger(Logger):
    def __init__(self, log_path):
        self._log_path = os.path.join(log_path, 'tboard') # TODO make this more generic
        self._make_dir()
        self._init_writer()
        global global_step
        global_step = 0

    @property
    def log_path(self):
        return self._log_path
    
    def _init_writer(self):
        try:
            from tensorboardX import SummaryWriter
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f'{e}. Please install TensorboardX to log tensorboards')
        global writer 
        writer = SummaryWriter(self.log_path)
  
    def step(self, step=None):
        if step is None:
            global_step += 1
        else:
            global_step = step

    def update(self, step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None: 
                continue            
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            writer.add_scalar(k, v, global_step if step is None else step) 
    
    def flush(self):
        writer.flush()

    def log(self, step=None, *args, **kwargs):
        # NOTE Need to include args here too, even though that seems weird. Required by abstract method!
        self.update(step, **kwargs)
        self.flush()



if __name__ == '__main__':

    print('Hello')

    from vi_embeddings.model import VI
    model = VI(1000, 100)

    logger = DefaultLogger('test')

    logger.add_logger('params', ParameterLogger('test', model, ['detached_params', 'pruned_params']), update_interval=1)
    # logger.add_logger('tensorboard', TensorboardLogger('test'), callbacks=['train_loss', 'val_loss'], update_interval=1)
    logger.add_logger('file', FileLogger('test'), callbacks=['train_loss', 'val_loss'], update_interval=1)

    for i in range(10):

        train_loss = i
        val_loss = i

        log_dict = {'train_loss': i, 'val_loss': i}
        
        logger.log(**log_dict)




