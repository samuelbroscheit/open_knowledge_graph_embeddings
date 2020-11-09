import os
from typing import Dict

from openkge.index_mapper import IndexMapper
from preprocessing.pipeline_job import PipelineJob
from utils.map_open_dataset_to_ids import convert_datasets_with_entity_mention_annotations, save_id_to_tokens_map, save_to_file


class MapToIds(PipelineJob):
    """

    """

    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                f"data/versions/{opts.data_version_name}/train_data_basic.txt",
                f"data/versions/{opts.data_version_name}/train_data_simple.txt",
                f"data/versions/{opts.data_version_name}/train_data_thorough.txt",
                f"data/versions/{opts.data_version_name}/validation_data.txt",
                f"data/versions/{opts.data_version_name}/validation_data_linked.txt",
                f"data/versions/{opts.data_version_name}/validation_data_linked_no_mention.txt",
                f"data/versions/{opts.data_version_name}/test_data.txt",
            ],
            provides=[
                f"data/versions/{opts.data_version_name}/mapped_to_ids/",
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        mapped_to_ids_dir = f'data/versions/{self.opts.data_version_name}/mapped_to_ids'

        self.log("Loading train_data_basic.txt")
        with open(f'data/versions/{self.opts.data_version_name}/train_data_basic.txt') as train_data_basic:
            self.log("Loading train_data_simple.txt")
            with open(f'data/versions/{self.opts.data_version_name}/train_data_simple.txt') as train_data_simple:
                self.log("Loading train_data_thorough.txt")
                with open(f'data/versions/{self.opts.data_version_name}/train_data_thorough.txt') as train_data_thorough:
                    self.log("Loading validation_data.txt")
                    with open(f'data/versions/{self.opts.data_version_name}/validation_data.txt') as validation_data:
                        self.log("Loading validation_data_linked.txt")
                        with open(f'data/versions/{self.opts.data_version_name}/validation_data_linked.txt') as validation_data_linked:
                            self.log("Loading test_data.txt")
                            with open(f'data/versions/{self.opts.data_version_name}/test_data.txt') as test_data:
                                mention_index_mapper_m = IndexMapper(segment=True)
                                relation_index_mapper_m = IndexMapper(segment=True)

                                # Argument to 'train' is the file from which the known
                                # tokens during training are collected. The collected
                                # tokens are then applied to all the files in
                                # 'valid_and_tests' (takes also other training
                                # data)
                                train_data_thorough_converted, (
                                    train_data_basic_converted,
                                    train_data_simple_converted,
                                    valid_converted,
                                    valid_linked_converted,
                                    test_converted
                                ), \
                                mention_id_token_ids_map, \
                                relation_id_token_ids_map = convert_datasets_with_entity_mention_annotations(
                                    train=train_data_thorough.readlines(),
                                    others_train=[
                                        train_data_basic.readlines(),
                                        train_data_simple.readlines(),
                                    ],
                                    valid_and_test=[
                                        validation_data.readlines(),
                                        validation_data_linked.readlines(),
                                        test_data.readlines()
                                    ],
                                    subj_index_mapper=mention_index_mapper_m,
                                    obj_index_mapper=mention_index_mapper_m,
                                    rel_index_mapper=relation_index_mapper_m,
                                    segment=True,
                                    collect_mention_vocab_also_from_others=True,

                                )

                                if not os.path.exists(mapped_to_ids_dir):
                                    os.makedirs(mapped_to_ids_dir)

                                mention_index_mapper_m.save_vocab(os.path.join(mapped_to_ids_dir, 'entity'))
                                relation_index_mapper_m.save_vocab(os.path.join(mapped_to_ids_dir, 'relation'))

                                save_id_to_tokens_map(mapped_to_ids_dir, 'entity', mention_id_token_ids_map)
                                save_id_to_tokens_map(mapped_to_ids_dir, 'relation', relation_id_token_ids_map)

                                save_to_file(mapped_to_ids_dir, 'train_data_basic.txt', train_data_basic_converted)
                                save_to_file(mapped_to_ids_dir, 'train_data_simple.txt', train_data_simple_converted)
                                save_to_file(mapped_to_ids_dir, 'train_data_thorough.txt', train_data_thorough_converted)
                                save_to_file(mapped_to_ids_dir, 'validation_data_all.txt', valid_converted)
                                save_to_file(mapped_to_ids_dir, 'validation_data_linked.txt', valid_linked_converted)
                                save_to_file(mapped_to_ids_dir, 'test_data.txt', test_converted)

        # with open(f"data/versions/{self.opts.data_version_name}/elasticsearch_index_created", "w",) as f:
        #     f.writelines(["SUCCESS"])
