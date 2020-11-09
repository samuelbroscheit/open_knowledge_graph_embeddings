import os
import subprocess
import torch
from ast import literal_eval

from torch.backends import cudnn

from openkge.dataset import Datasets, OneToNMentionRelationDataset


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])


def resume_checkpoint(args):
    checkpoint = torch.load(args["resume"], map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu', 'cuda:2': 'cpu', 'cuda:3': 'cpu', })
    resume_args = checkpoint['config']
    for k, v in resume_args.items():
        if k in [
            'train',
            'evaluate',
            'evaluate_scores_file',
            'evaluate_on_validation',
            'devices',
            'resume',
            'reset_optimizer',
            'resume_freeze',
            'resume_filter',
            'patience_epochs',
            'patience_metric_treshold',
            'patience_metric_change',
            'eval_freq',
            'eval_epoch_freq',
            'save_freq',
            'print_freq',
            'save_freq',
            'no_cuda',
            'epochs',
        ]:
            continue

        elif k in [
            'train_data_config',
            'val_data_config',
            'test_data_config',
            'training_dataset',
            'validation_dataset',
            'test_dataset',
            'model_select_metric',
            'results_dir',
            'experiment_dir',
        ] and args[k] is not None and (isinstance(args[k], str) and len(args[k]) > 0 or not isinstance(args[k], str)):
            continue

        else:
            args[k] = v
    return args, checkpoint


def setup_devices(args):
    main_gpu = 0
    if not args["no_cuda"]:
        if isinstance(args["devices"], tuple):
            main_gpu = args["devices"][0]
        elif isinstance(args["devices"], int):
            main_gpu = args["devices"]
        elif isinstance(args["devices"], dict):
            main_gpu = args["devices"].get('input', 0)
        torch.cuda.set_device(main_gpu)
        cudnn.benchmark = True
    else:
        main_gpu = 'cpu'
    return main_gpu


def setup_dirs(args, hyper_setting, time_stamp=None):
    if args["experiment_dir"] is None:
        if time_stamp is not None:
            args["experiment_dir"] = '{}-{}-{}'.format(os.path.basename(args["config"]), hyper_setting, time_stamp)
        else:
            args["experiment_dir"] = '{}-{}'.format(os.path.basename(args["config"]), hyper_setting)

    save_path = os.path.join(args["results_dir"], args["experiment_dir"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path


def setup_dataset_classes(args):
    if args["training_dataset_class"] is None:
        training_dataset = getattr(Datasets, args["dataset_class"])
    else:
        training_dataset = getattr(Datasets, args["training_dataset_class"])

    if args["validation_dataset_class"] is None:
        validation_dataset = getattr(Datasets, args["dataset_class"])
    else:
        validation_dataset = getattr(Datasets, args["validation_dataset_class"])

    if args["test_dataset_class"] is None:
        test_dataset = getattr(Datasets, args["dataset_class"])
    else:
        test_dataset = getattr(Datasets, args["test_dataset_class"])

    return training_dataset, validation_dataset, test_dataset


def setup_dataset(args, dataset_clazz, data_config, main_gpu, is_training_data):
    if not isinstance(data_config, dict):
        data_config = literal_eval(data_config)
    if 'batch_size' not in data_config or data_config['batch_size'] is None:
        data_config['batch_size'] = args["batch_size"]
    if 'device' not in data_config or args["resume"]:
        data_config['device'] = main_gpu if not args["no_cuda"] or 'cuda' in data_config and data_config['cuda'] or args["resume"] else 'cpu'
    dataset = dataset_clazz(dataset_dir=args["dataset_dir"], is_training_data=is_training_data, **data_config, **args["experiment_settings"])
    return dataset
