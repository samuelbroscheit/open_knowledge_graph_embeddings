import logging
import os
import random
from datetime import datetime

import numpy
import pandas
import torch.optim
from torch.nn import BCEWithLogitsLoss, KLDivLoss, CrossEntropyLoss

from openkge.model import *
from openkge.options import preprocess_args_and_create_parser
from openkge.trainer import Trainer
from openkge.utils import resume_checkpoint, setup_devices, setup_dirs, setup_dataset_classes, setup_dataset
from utils.log import setup_logging

def main(args, hyper_setting='', time_stamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')):

    print(args)

    # setup device
    main_gpu = setup_devices(args)

    # setup resuming checkpoint
    checkpoint = None
    if args["resume"]:
        args, checkpoint = resume_checkpoint(args)

    # setup output directories
    save_path = setup_dirs(args, time_stamp=time_stamp, hyper_setting=hyper_setting)

    # setup logger
    root_logger = setup_logging(os.path.join(save_path, 'log_%s.txt' % time_stamp))

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    # seed RNGs
    if args["seed"] > 0:
        random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        numpy.manual_seed(args["seed"])

    training_dataset_class, validation_dataset_class, test_dataset_class = setup_dataset_classes(args)

    train_data = setup_dataset(args, training_dataset_class, args["train_data_config"], main_gpu, is_training_data=True)
    valid_data = setup_dataset(args, validation_dataset_class, args["val_data_config"], main_gpu, is_training_data=False)
    test_data = setup_dataset(args, test_dataset_class, args["test_data_config"], main_gpu, is_training_data=False)

    # check if we are evaluating on validation or test
    if args["evaluate"]:
        args["train"] = False
    if args["evaluate_on_validation"]:
        evaluation_data = valid_data
    else:
        evaluation_data = test_data

    # merge train, valid and test for filtering
    evaluation_data.merge_all_splits_triples(
        dataset_dir=args["dataset_dir"],
        train_input_file=train_data.input_file_name,
        valid_input_file=valid_data.input_file_name,
        test_input_file=test_data.input_file_name,
    )

    # create/load training data in tensors
    train_data.create_data_tensors(
        dataset_dir=args["dataset_dir"],
        train_input_file=train_data.input_file_name,
        valid_input_file=valid_data.input_file_name,
        test_input_file=test_data.input_file_name,
    )

    # create/load evaluation data in tensors
    evaluation_data.create_data_tensors(
        dataset_dir=args["dataset_dir"],
        train_input_file=train_data.input_file_name,
        valid_input_file=valid_data.input_file_name,
        test_input_file=test_data.input_file_name,
    )

    # optimization regime and model config

    # now initialize the model

    args["model_config"]['train_data'] = train_data.get_dataset_meta_dict()

    model = getattr(Models, args["model"])(**args["model_config"])

    #FP16 precision
    model.half()
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
            
    logging.info(model)

    # define data loaders

    train_loader = train_data.get_loader(
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    #FP16
    for batch_size, inputs in enumerate(train_loader):
        inputs = inputs.to(device).half()

    val_loader = evaluation_data.get_loader(
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    #FP16
    for batch_size, inputs in enumerate(val_loader):
        inputs = inputs.to(device).half()

    # create trainer

    trainer = Trainer(
        loss=
        CrossEntropyLoss(reduction='sum') if args["experiment_settings"]['loss'] == 'ce' else
        BCEWithLogitsLoss(reduction='sum') if args["experiment_settings"]['loss'] == 'bce' else
        KLDivLoss(reduction='sum') if args["experiment_settings"]['loss'] == 'kl' else None,
        args=args,
        model=model,
        train_dataset=train_data,
        validation_dataset=evaluation_data,
        train_loader=train_loader,
        save_path=save_path,
    )

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    if not args["no_cuda"]:
        model.cuda()

    # optionally resume from a checkpoint
    if args["resume"]:
            checkpoint_file = args["resume"]
            if os.path.isfile(checkpoint_file):
                trainer.load(checkpoint_file,
                             resume_filter=args["resume_filter"],
                             freeze_param=args["resume_freeze"],
                             reset_optimizer=args["reset_optimizer"],
                             weight_map=None,
                             checkpoint=checkpoint,
                             dont_load_optimizer=args["evaluate"],
                             )
            else:
                logging.error("no checkpoint found at '%s'", args["resume"])

    if args["train"]:

        try:
            while trainer.epoch < args["epochs"]:
                # train for one epoch
                trainer.run(train_loader, val_loader)
                if trainer.terminate:
                    break
        except KeyboardInterrupt:
            for handler in root_logger.handlers: handler.flush()
            pass

    elif args["evaluate"]:

        validation_criterion_results = trainer.evaluate(val_loader, save=False)

        if args["evaluate_scores_file"]:
            with_header_or_not = not os.path.exists(args["evaluate_scores_file"])
            with open(args["evaluate_scores_file"], 'a') as f:
                optimization_config = args["optimization_config"]

                if with_header_or_not:
                    with_header_or_not = [
                        'checkpoint_path',
                        'checkpoint',
                        'batch_size',
                        'entity_slot_size',
                        'relation_slot_size',
                        'dropout',
                        'input_dropout',
                        'relation_dropout',
                        'relation_input_dropout',
                        'model',
                        'train_data',
                        'valid_data',
                        'sparse',
                        'lr',
                        'weight_decay',
                        'epoch',
                        'model_select_criterion',
                        'loss',
                        'mrr',
                        'mr',
                        'h50',
                        'h10',
                        'h3',
                        'h1',
                    ]

                experiment_config = [
                    os.path.basename(os.path.dirname(args["resume"])),
                    os.path.basename(args["resume"]),
                    args["batch_size"],
                    args["model_config"].get('entity_slot_size', '-'),
                    args["model_config"].get('relation_slot_size', '-'),
                    args["model_config"].get('dropout', '-'),
                    args["model_config"].get('input_dropout', '-'),
                    args["model_config"].get('relation_dropout', '-'),
                    args["model_config"].get('relation_input_dropout', '-'),
                    args["model"],
                    args["train_data_config"].get('input_file', '-'),
                    args["val_data_config"].get('input_file', '-'),
                    args["model_config"].get('sparse', '-'),
                    optimization_config[0].get('lr', '-') if isinstance(optimization_config, list) else optimization_config.get('lr', '-') if isinstance(optimization_config, dict) else '-',
                    optimization_config[0].get('weight_decay', '-') if isinstance(optimization_config, list) else optimization_config.get('weight_decay', '-') if isinstance(optimization_config, dict) else '-',
                ]
                # if hasattr(trainer.model, 'sparsity'):
                #     experiment_config.append(trainer.model.sparsity)

                pandas.DataFrame([experiment_config + [
                    trainer.epoch,
                    validation_criterion_results[args["model_select_metric"][0]],
                    validation_criterion_results['loss'],
                    validation_criterion_results['mrr'],
                    validation_criterion_results['mr'],
                    validation_criterion_results['h50'],
                    validation_criterion_results['h10'],
                    validation_criterion_results['h3'],
                    validation_criterion_results['h1']]
                ]).to_csv(f, header=with_header_or_not)


    for handler in root_logger.handlers: handler.flush()

    return trainer.best_validation_results[args["model_select_metric"][0]], \
           trainer.best_validation_results['loss'], \
           trainer.best_validation_results['mrr'], \
           trainer.best_validation_results['mr'], \
           trainer.best_validation_results['h50'], \
           trainer.best_validation_results['h10'], \
           trainer.best_validation_results['h3'], \
           trainer.best_validation_results['h1'],


if __name__ == '__main__':
    args = preprocess_args_and_create_parser()
    if args:
        main(args)
