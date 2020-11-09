import distutils.util
import argparse
import sys
from pathlib import Path

import yaml


def preprocess_args_and_create_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--results_dir', help='results dir')
    parser.add_argument('--experiment_dir', help='Name for this experiment (will be a timestamp if empty)')

    parser.add_argument('--dataset_class', )
    parser.add_argument('--training_dataset_class', )
    parser.add_argument('--validation_dataset_class', )
    parser.add_argument('--test_dataset_class', )
    parser.add_argument('--dataset_dir', help='dataset dir')
    parser.add_argument('--train_data_config', help='data configuration')
    parser.add_argument('--val_data_config', help='data configuration')
    parser.add_argument('--test_data_config', help='data configuration')

    parser.add_argument('--model', )
    parser.add_argument('--model_config', help='Architecture configuration')
    parser.add_argument('--devices', )
    parser.add_argument('--batch_size', type=int, help='Mini-batch size')
    parser.add_argument('--epochs', type=int, help='Number of total epochs to run')
    parser.add_argument('--start_epoch', type=int, help='Manual set first epoch number (useful on restarts)')
    parser.add_argument('--experiment_settings', help='Experiment settings')

    parser.add_argument('--optimization_config', type=str, help='Optimization')
    parser.add_argument('--lr_scheduler_config', type=str, help='Learning rate scheduler regime')
    parser.add_argument('--grad_clip', type=float, help='Maximum grad norm value')
    parser.add_argument('--bce_label_smoothing', type=float, help='Label smoothing coefficient')

    parser.add_argument('--print_freq', type=int, help='Print frequency')
    parser.add_argument('--save_freq', type=int, help='Save frequency')
    parser.add_argument('--save_epoch_freq', type=int, help='Save frequency')
    parser.add_argument('--eval_freq', type=int, help='Evaluation frequency')
    parser.add_argument('--eval_epoch_freq', type=int, help='Evaluation frequency')
    parser.add_argument('--patience_epochs', type=int, help='')
    parser.add_argument('--patience_metric_min_treshold', type=float, help='')
    parser.add_argument('--patience_metric_max_treshold', type=float, help='')
    parser.add_argument('--patience_metric_change', type=float, help='')
    parser.add_argument('--model_select_metric', action='append', help='Evaluation metric to use for model selection [min_ppl|max_bleu|min_loss]')

    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--no_cuda', type=lambda x:bool(distutils.util.strtobool(x)), help='Train model')
    parser.add_argument('--train', type=lambda x:bool(distutils.util.strtobool(x)), help='Train model')
    parser.add_argument('--evaluate', type=lambda x:bool(distutils.util.strtobool(x)), help='Evaluate model')
    parser.add_argument('--evaluate_on_validation', type=lambda x:bool(distutils.util.strtobool(x)), help='Evaluate model')
    parser.add_argument('--evaluate_scores_file', type=str, help="File to store csv results")

    parser.add_argument('--resume', type=str, help='Path to checkpoint (default: none)')
    parser.add_argument('--resume_load_args', type=lambda x:bool(distutils.util.strtobool(x)), help='Path to checkpoint (default: none)')
    parser.add_argument('--resume_filter', action="append", help='List of weight names that should be filtered out to resume')
    parser.add_argument('--resume_freeze', action="append", help='Freeze the resumed parameters (either True for all or dict)')
    parser.add_argument('--reset_optimizer', type=lambda x:bool(distutils.util.strtobool(x)), help='Use the checkpoint parameters as pretraining, else they are used as warm restart')

    parser.add_argument('--log_predictions', action='store_true',)

    if len(sys.argv) == 1:
        sys.stderr.write(f"Usage: {__file__} [CONFIG.yaml] [MORE OPTIONS]")
        return False
    elif len(sys.argv) > 1:
        parsed_args = None
        if not sys.argv[1].lower().endswith(".yaml"):
            file_name, yaml_file_name, args = sys.argv[0], str(Path(__file__).parent.absolute()) + "/default.yaml", sys.argv[1:]
        else:
            file_name, yaml_file_name, args = sys.argv[0], sys.argv[1], sys.argv[2:]
        if len(args) > 0:
            sys.argv = [file_name] + args
            parsed_args = parser.parse_args()

        with open(yaml_file_name) as file:
            try:
                config = yaml.full_load(file)
            except:
                sys.stderr.write(f"Usage: {__file__} CONFIG.yaml [MORE OPTIONS]")
                return False

        if parsed_args:
            for k, v in parsed_args.__dict__.items():
                if k in config and v is not None:
                    config[k] = v

        config["config"] = yaml_file_name
        return config

