from absl import flags
from flax import optim
import flax
from gnp.optimizer.Adam_class import AdamOptimizer
from gnp.optimizer.SGD_class import SGDOptimizer

FLAGS = flags.FLAGS


def get_optimizer(params: flax.nn.Model,
                 learning_rate: float,
                 ):

    if FLAGS.config.opt.opt_type == "SGD":
        optimizer_def = SGDOptimizer(learning_rate=learning_rate, **FLAGS.config.opt.opt_params)
    elif FLAGS.config.opt.opt_type == "Momentum":
        optimizer_def = optim.Momentum(learning_rate=learning_rate, **FLAGS.config.opt.opt_params)
    elif FLAGS.config.opt.opt_type == "Adam":
        optimizer_def = AdamOptimizer(learning_rate=learning_rate, **FLAGS.config.opt.opt_params)

    optimizer = optimizer_def.create(params)
    return optimizer