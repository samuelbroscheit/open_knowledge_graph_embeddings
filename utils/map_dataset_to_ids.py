import os
from collections import OrderedDict

from operator import itemgetter
from os import makedirs
from typing import List

from openkge.index_mapper import IndexMapper, UNK


def save_to_file(dataset_dir, file_name, triples):

    if not os.path.exists(dataset_dir):
        makedirs(dataset_dir)

    with open(os.path.join(dataset_dir, file_name), 'w') as f:
        f.writelines(['\t'.join([' '.join([str(i) for i in triple[slot]]) for slot in [0,1,2,0,2]]) + '\n' for triple in triples])

def save_id_to_tokens_map(dataset_dir, item_name, mappings, suffix='_id_tokens_ids_map', file_type_suffix='.txt'):
    if not os.path.exists(dataset_dir):
        makedirs(dataset_dir)
    with open(os.path.join(dataset_dir, item_name + suffix + file_type_suffix), 'w', encoding='UTF-8') as f:
        f.write("# {} id\ttokens\t\n".format(item_name))
        for mapping in sorted(mappings.items(), key=lambda x: x[0]):
            id, tokens = mapping
            f.write("{0}\t{1}\n".format(id, ' '.join([str(t) for t in tokens])))


def convert_datasets(train: List,
                     valid: List,
                     test: List,
                     subj_index_mapper: IndexMapper,
                     obj_index_mapper: IndexMapper,
                     rel_index_mapper: IndexMapper,
                     triple_format_parser=lambda x: x.strip().split('\t'),
                     subj_slot=0,
                     rel_slot=1,
                     obj_slot=2,
                     filter_unseen=False,
                     filter_func=None,
                     segment=False,
                     ):

    if not filter_func:
        max_number_of_unknowns = 0
        filter_func = lambda i: sum([1 if sum([1 if UNK != k else 0 for k in j]) <= max_number_of_unknowns else 0 for j in i]) == 0

    idx_mappers = [subj_index_mapper, rel_index_mapper, obj_index_mapper]

    for idx_mapper in idx_mappers: idx_mapper.init_vocab()

    # collect vocab from train data

    datasets_for_collecting_vocab = [train]

    if not filter_unseen:
        datasets_for_collecting_vocab.extend([valid, test])

    for data in datasets_for_collecting_vocab:
        for x in data:
            x = triple_format_parser(x)
            subj_index_mapper.collect_vocab(x[subj_slot])
            rel_index_mapper.collect_vocab(x[rel_slot])
            obj_index_mapper.collect_vocab(x[obj_slot])

    for idx_mapper in idx_mappers: idx_mapper.finalize_vocab()

    def convert_data_to_idx(data):
        return [
            (list(zip(*(subj_index_mapper.toidx(s),
                        rel_index_mapper.toidx(r),
                        obj_index_mapper.toidx(o))
                      )
                  )
             )
            for s,r,o in map(triple_format_parser, data)
        ]

    # apply vocab to all data

    train_converted = convert_data_to_idx(train)
    valid_converted = convert_data_to_idx(valid)
    test_converted = convert_data_to_idx(test)

    if segment:
        entity_id_token_ids_map = OrderedDict()
        relation_id_token_ids_map = OrderedDict()
        for triple__id_and_segmented in train_converted + valid_converted + test_converted:
            triple, triple_segmented = triple__id_and_segmented
            # if triple[subj_slot][0] == 1 or triple[obj_slot][0] == 1:
            #     print(triple)
            #     print(triple_segmented)
            entity_id_token_ids_map[triple[subj_slot][0]] = triple_segmented[subj_slot]
            relation_id_token_ids_map[triple[rel_slot][0]] = triple_segmented[rel_slot]
            entity_id_token_ids_map[triple[obj_slot][0]] = triple_segmented[obj_slot]
        return filter(filter_func, map(itemgetter(0), train_converted)),\
               filter(filter_func, map(itemgetter(0), valid_converted)),\
               filter(filter_func, map(itemgetter(0), test_converted)),\
               entity_id_token_ids_map,\
               relation_id_token_ids_map
    else:
        return filter(filter_func, map(itemgetter(0), train_converted)),\
               filter(filter_func, map(itemgetter(0), valid_converted)),\
               filter(filter_func, map(itemgetter(0), test_converted))





train = """B O|works in|N Y
O|works in|W
G B|is president of|USA
A M|is chancelor of|DE
G H|is chancelor of|DE
M|is president of|F
A M|works in|Berlin
G H|works in|Bonn
Bonn|is in|DE
Berlin|is in|DE
N Y|is in|USA
W|is in|USA"""

valid = """B O|works in|USA
M|is chancelor of|DE
G H|works in|DE"""

test = """O|works in|USA
G B|works in|USA
A M|works in|DE"""

if __name__ == "__main__":

    entity_segment_index_mapper = IndexMapper(segment=True)
    relation_segment_index_mapper = IndexMapper(segment=True)

    tr, vl, te, eis_map, ris_map = convert_datasets(
        train=train.split('\n'),
        valid=valid.split('\n'),
        test=test.split('\n'),
        subj_index_mapper=entity_segment_index_mapper,
        obj_index_mapper=entity_segment_index_mapper,
        rel_index_mapper=relation_segment_index_mapper,
        triple_format_parser=lambda x: x.strip().split('|'),
        segment=True,
    )

    print(list(tr), list(vl), list(te))

    print(sorted(eis_map.items(), key=lambda x: x[0]))
    print(sorted(ris_map.items(), key=lambda x: x[0]))

    print(entity_segment_index_mapper.item2idx)
    print(entity_segment_index_mapper.item2segmentidx)

    print(relation_segment_index_mapper.item2idx)
    print(relation_segment_index_mapper.item2segmentidx)

