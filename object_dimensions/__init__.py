from .core.model import VariationalEmbedding, DeterministicEmbedding
from .core.priors import SpikeSlabPrior, LogGaussianPrior
from .core.pruning import NormalDimensionPruning, LogNormalDimensionPruning
from .core.engine import EmbeddingTrainer
from .core.dataset import TripletDataset, build_triplet_dataset
from .logging.loggers import DeepEmbeddingLogger

# from .extraction.extract_model_features import extract_features
# from .extraction.sampler import Sampler


from .core import priors, engine, dataset, pruning, model
from .utils import utils
