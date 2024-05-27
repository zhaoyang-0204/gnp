from gnp.models import load_model
import os
from absl import flags
from absl import app
from ml_collections.config_flags import config_flags
from gnp.training import utli
import jax
from gnp.optimizer.get_optimizer import get_optimizer, manage_optimizer_config
import functools
from gnp.training import training_core
from gnp.ds_pipeline.get_dataset import get_dataset_pipeline
from flax import jax_utils
import glob

FLAGS = flags.FLAGS
WORK_DIR = flags.DEFINE_string('working_dir', None,
                               'Directory to store configs, logs and checkpoints.')
config_flags.DEFINE_config_file(
    "config",
    None,
    'File path to the training hyperparameter configuration.'
)
flags.mark_flags_as_required(["config", "working_dir"])

base_path = "/raid/zy/models/ensemble_test/0/ResNet18/cifar100/basic_none/lr_0.1/bs_256/l2reg_0.001/grad_clip_5.0/opt_Momentum/warmup_epochs_0/r_0.1/alpha_1.0/epoch_200/"

def read_path():
    paths = sorted(glob.glob(os.path.join(base_path, "*")))
    paths = [os.path.join(item, "checkpoints") for item in paths]
    return paths

def main(_):
    manage_optimizer_config()
    # checkpoint_dir = "/raid/zy/models/ensemble_test/0/ResNet18/cifar100/basic_none/lr_0.1/bs_256/l2reg_0.001/grad_clip_5.0/opt_Momentum/warmup_epochs_0/r_0.1/alpha_1.0/epoch_200/seeds_14_0"
    # checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
    ds = get_dataset_pipeline()
    model, variables = load_model.get_model(FLAGS.config.model.model_name,
                                            FLAGS.config.batch_size,
                                            FLAGS.config.dataset.image_size,
                                            FLAGS.config.dataset.num_classes,
                                            FLAGS.config.dataset.num_channels,
                                            FLAGS.config.init_seeds)
    
    optimizer = get_optimizer(0.)
    optimizer = optimizer(learning_rate=0, momentum=0.9, **FLAGS.config.opt.opt_params)
    origin_opt_state = optimizer.init(variables['params'])
    origin_variables = variables
    paths = read_path()
    for i, checkpoint_dir in enumerate(paths[1:2]):
        opt_state, variables, epoch_last_checkpoint = utli.restore_checkpoint(
            origin_opt_state, origin_variables, checkpoint_dir)
        if i == 0:
            opt_state_mean = opt_state
            variables_mean = variables
        else:
            opt_state_mean = jax.tree_map(lambda a, b : (a + b) / 2, opt_state_mean, opt_state)
            variables_mean = jax.tree_map(lambda a, b : (a + b) / 2, variables_mean, variables)

    opt_state = jax_utils.replicate(opt_state_mean)
    variables = jax_utils.replicate(variables_mean)
    prng_key = jax.random.PRNGKey(FLAGS.config.seeds)
    pmapped_eval_step = jax.pmap(
        functools.partial(
            training_core.eval_step,
            apply_fn = model.apply,),
         axis_name='batch')
    
    test_ds = ds.get_test()
    test_metrics = training_core.eval_on_dataset(
        variables, test_ds, pmapped_eval_step)
    print(epoch_last_checkpoint)
    print(test_metrics)

if __name__ == "__main__":
    app.run(main)
