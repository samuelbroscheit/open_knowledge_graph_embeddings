import gzip
import os

from openkge.index_mapper import IndexMapper
from utils.map_dataset_to_ids import convert_datasets, save_to_file, save_id_to_tokens_map

if __name__ == "__main__":

    with gzip.open('mid2name.tsv.gz', 'rb') as f:
        mid2name = {mid_name[0].decode():[m.decode() for m in mid_name[1:]] for mid_name in [line.split() for line in f.readlines()]}

    entity_index_mapper = IndexMapper(
        segment=True,
        segment_func=lambda line: mid2name.get(line, [line])
    )

    relation_index_mapper = IndexMapper(
        segment=True,
        segment_func=lambda line: line.replace('/', ' / ').replace('.', ' . ').replace('_', ' ').split(),
    )

    with open('train.txt') as train:
        with open('valid.txt') as valid:
            with open('test.txt') as test:
                train_converted, \
                valid_converted, \
                test_converted, \
                entity_id_token_ids_map, \
                relation_id_token_ids_map = convert_datasets(
                    train=train.readlines(),
                    valid=valid.readlines(),
                    test=test.readlines(),
                    subj_index_mapper=entity_index_mapper,
                    obj_index_mapper=entity_index_mapper,
                    rel_index_mapper=relation_index_mapper,
                    triple_format_parser=lambda x: x.strip().split(),
                    segment=True,
                )

                if not os.path.exists('mapped_to_ids'):
                    os.makedirs('mapped_to_ids')

                entity_index_mapper.save_vocab(os.path.join('mapped_to_ids', 'entity'))
                relation_index_mapper.save_vocab(os.path.join('mapped_to_ids', 'relation'))

                save_id_to_tokens_map('mapped_to_ids', 'entity', entity_id_token_ids_map)
                save_id_to_tokens_map('mapped_to_ids', 'relation', relation_id_token_ids_map)

                save_to_file('mapped_to_ids', 'train.txt', train_converted)
                save_to_file('mapped_to_ids', 'valid.txt', valid_converted)
                save_to_file('mapped_to_ids', 'test.txt', test_converted)

