import concurrent
import itertools
import multiprocessing
import os
from collections import OrderedDict, Counter

from operator import itemgetter
from os import makedirs
from typing import List

from tqdm import tqdm

from openkge.index_mapper import IndexMapper, UNK


def save_to_file(dataset_dir, file_name, triples):

    if not os.path.exists(dataset_dir):
        makedirs(dataset_dir)

    with open(os.path.join(dataset_dir, file_name), 'w') as f:
        f.writelines(['\t'.join([' '.join([str(i) for i in triple[slot]]) for slot in [0,1,2]]) + '\t' + '\t'.join([' '.join([str(i[0]) for i in triple[slot]]) for slot in [3,4]]) + '\n' for triple in triples])

def save_id_to_tokens_map(dataset_dir, item_name, mappings, suffix='_id_tokens_ids_map', file_type_suffix='.txt'):
    if not os.path.exists(dataset_dir):
        makedirs(dataset_dir)
    with open(os.path.join(dataset_dir, item_name + suffix + file_type_suffix), 'w', encoding='UTF-8') as f:
        f.write("# {} id\ttokens\t\n".format(item_name))
        for mapping in sorted(mappings.items(), key=lambda x: x[0]):
            id, tokens = mapping
            f.write("{0}\t{1}\n".format(id, ' '.join([str(t) for t in tokens])))


def convert_datasets_with_entity_annotations(train: List,
                     valid: List,
                     test: List,
                     subj_index_mapper: IndexMapper,
                     obj_index_mapper: IndexMapper,
                     rel_index_mapper: IndexMapper,
                     subj_entity_index_mapper: IndexMapper,
                     obj_entity_index_mapper: IndexMapper,
                     triple_format_parser=lambda x: x.strip().split('\t'),
                     subj_slot=0,
                     rel_slot=1,
                     obj_slot=2,
                     subj_entity_slot=3,
                     obj_entity_slot=4,
                     filter_func=None,
                     segment=False,
                     ):

    filter_func = lambda i: True

    # if not filter_func:
    #     max_number_of_unknowns = 0
    #     filter_func = lambda i: sum([1 if sum([1 if UNK != k else 0 for k in j]) <= max_number_of_unknowns else 0 for j in i]) == 0

    idx_mappers = [subj_index_mapper, rel_index_mapper, obj_index_mapper, subj_entity_index_mapper, obj_entity_index_mapper]

    for idx_mapper in idx_mappers: idx_mapper.init_vocab()

    # collect vocab from train data

    for x in train:
        x = triple_format_parser(x)
        subj_index_mapper.collect_vocab(x[subj_slot])
        rel_index_mapper.collect_vocab(x[rel_slot])
        obj_index_mapper.collect_vocab(x[obj_slot])
        subj_entity_index_mapper.collect_vocab(x[subj_entity_slot])
        obj_entity_index_mapper.collect_vocab(x[obj_entity_slot])

    for idx_mapper in idx_mappers: idx_mapper.finalize_vocab()

    def convert_data_to_idx(data):
        return [
            (list(zip(*(
                subj_index_mapper.toidx(s),
                rel_index_mapper.toidx(r),
                obj_index_mapper.toidx(o),
                subj_entity_index_mapper.toidx(se),
                obj_entity_index_mapper.toidx(oe),
            ))))
            for s,r,o,se,oe in map(triple_format_parser, data)
        ]

    # apply vocab to all data

    train_converted = convert_data_to_idx(train)
    valid_converted = convert_data_to_idx(valid)
    test_converted = convert_data_to_idx(test)

    entity_id_mention_ids_map = OrderedDict()
    for triple__id_and_segmented in train_converted + valid_converted + test_converted:
        triple, _ = triple__id_and_segmented

        if triple[subj_entity_slot][0] not in entity_id_mention_ids_map:
            entity_id_mention_ids_map[triple[subj_entity_slot][0]] = Counter()
        entity_id_mention_ids_map[triple[subj_entity_slot][0]][triple[subj_slot][0]] += 1

        if triple[obj_entity_slot][0] not in entity_id_mention_ids_map:
            entity_id_mention_ids_map[triple[obj_entity_slot][0]] = Counter()
        entity_id_mention_ids_map[triple[obj_entity_slot][0]][triple[obj_slot][0]] += 1

    if segment:
        mention_id_token_ids_map = OrderedDict()
        relation_id_token_ids_map = OrderedDict()
        for triple__id_and_segmented in train_converted + valid_converted + test_converted:
            triple, triple_segmented = triple__id_and_segmented
            mention_id_token_ids_map[triple[subj_slot][0]] = triple_segmented[subj_slot]
            relation_id_token_ids_map[triple[rel_slot][0]] = triple_segmented[rel_slot]
            mention_id_token_ids_map[triple[obj_slot][0]] = triple_segmented[obj_slot]
        return filter(filter_func, map(itemgetter(0), train_converted)),\
               filter(filter_func, map(itemgetter(0), valid_converted)),\
               filter(filter_func, map(itemgetter(0), test_converted)),\
               mention_id_token_ids_map,\
               relation_id_token_ids_map, \
               entity_id_mention_ids_map
    else:
        return filter(filter_func, map(itemgetter(0), train_converted)),\
               filter(filter_func, map(itemgetter(0), valid_converted)),\
               filter(filter_func, map(itemgetter(0), test_converted)), \
               entity_id_mention_ids_map


class Worker(multiprocessing.Process):

    def __init__(self, in_queue,
                 out_queue,
                 subj_index_mapper,
                 rel_index_mapper,
                 obj_index_mapper,
                 mention_format_parser,
                 triple_format_parser
                 ):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.subj_index_mapper = subj_index_mapper
        self.rel_index_mapper = rel_index_mapper
        self.obj_index_mapper = obj_index_mapper
        self.mention_format_parser = mention_format_parser
        self.triple_format_parser = triple_format_parser

    def run(self):
        for next_item in iter(self.in_queue.get, None):
            file_name = next_item
            self.out_queue.put((self.extract_data(next_item), file_name))

    def extract_data(self, data):
        return [
            (list(zip(*(
                self.subj_index_mapper.toidx(s),
                self.rel_index_mapper.toidx(r),
                self.obj_index_mapper.toidx(o),
                list(zip(*((self.subj_index_mapper.toidx(sem, segment=False) for sem in self.mention_format_parser(sems))))),
                list(zip(*((self.obj_index_mapper.toidx(oem, segment=False) for oem in self.mention_format_parser(oems))))),
            ))))
            for s, r, o, sems, oems in tqdm(map(self.triple_format_parser, data))
        ]

def convert_datasets_with_entity_mention_annotations(train: List,
                                                     subj_index_mapper: IndexMapper,
                                                     obj_index_mapper: IndexMapper,
                                                     rel_index_mapper: IndexMapper,
                                                     others_train: List[List] = [],
                                                     valid_and_test: List[List] = [],
                                                     triple_format_parser=lambda x: x.strip().split('\t'),
                                                     mention_format_parser=lambda x: [y.strip() for y in x.strip().split('|||')],
                                                     subj_slot=0,
                                                     rel_slot=1,
                                                     obj_slot=2,
                                                     subj_entity_slot=3,
                                                     obj_entity_slot=4,
                                                     filter_func=None,
                                                     segment=False,
                                                     collect_mention_vocab_also_from_others=False,
                                                     ):

    idx_mappers = [subj_index_mapper, rel_index_mapper, obj_index_mapper]

    for idx_mapper in idx_mappers: idx_mapper.init_vocab()

    # collect vocab from train data

    print("Collect vocab from train data")
    for x in tqdm(train):
        x = triple_format_parser(x)
        subj_index_mapper.collect_vocab(x[subj_slot])
        rel_index_mapper.collect_vocab(x[rel_slot])
        obj_index_mapper.collect_vocab(x[obj_slot])
        # for sem in mention_format_parser(x[subj_entity_slot]): subj_index_mapper.collect_vocab(sem)
        # for oem in mention_format_parser(x[obj_entity_slot]): obj_index_mapper.collect_vocab(oem)

    # collect mentions vocab also from valid and test data

    print("Collect *mentions* vocab (not tokens) also from other train data")
    if collect_mention_vocab_also_from_others:
        for vt in others_train:
            for x in tqdm(vt):
                x = triple_format_parser(x)
                subj_index_mapper.collect_vocab(x[subj_slot], segment=False)
                rel_index_mapper.collect_vocab(x[rel_slot], segment=False)
                obj_index_mapper.collect_vocab(x[obj_slot], segment=False)

    for vt in valid_and_test:
        for x in tqdm(vt):
            x = triple_format_parser(x)
            subj_index_mapper.collect_vocab(x[subj_slot], segment=False)
            rel_index_mapper.collect_vocab(x[rel_slot], segment=False)
            obj_index_mapper.collect_vocab(x[obj_slot], segment=False)

    for idx_mapper in idx_mappers: idx_mapper.finalize_vocab()

    nr_of_workers = 20
    def convert_data_to_idx(data):

        in_queue = multiprocessing.Queue()
        out_queue = multiprocessing.Queue()

        workers = list()
        for id in range(nr_of_workers):
            worker = Worker(
                in_queue,
                out_queue,
                subj_index_mapper,
                rel_index_mapper,
                obj_index_mapper,
                mention_format_parser,
                triple_format_parser
            )
            worker.start()
            workers.append(worker)

        submitted_jobs = 0
        n = 10240
        for file_nr, tmp_input in enumerate(tqdm( [data[i:i + n] for i in range(0, len(data), n)]  )):
            submitted_jobs += 1
            in_queue.put((tmp_input))

        result = list()
        for _ in tqdm(range(submitted_jobs)):
            (tmp_result), in_file_name = out_queue.get()
            result.extend(tmp_result)

        # put the None into the queue so the loop in the run() function of the worker stops
        for worker in workers:
            in_queue.put(None)
            out_queue.put(None)

        # terminate the process
        for worker in workers:
            worker.join()

        return result

    # apply vocab to all data

    print("Apply mention vocab to all data")
    train_converted = convert_data_to_idx(train)

    others_train_converted = list()
    for other in others_train:
        others_train_converted.append(convert_data_to_idx(other))

    valid_and_test_converted = list()
    for vt in valid_and_test:
        valid_and_test_converted.append(convert_data_to_idx(vt))

    max_number_of_unknowns = 2/3
    classify_as_too_many_unknowns = lambda i: sum(map(lambda k: 1 if k == UNK else 0, i))/(len(i)-2) > max_number_of_unknowns

    if segment:

        print("Collect mention token map and filter")

        mention_id_token_ids_map = OrderedDict()
        relation_id_token_ids_map = OrderedDict()

        def collect_mentions_and_filter(data):
            result = list()
            for triple, triple_segmented in tqdm(data):
                # if triple[subj_slot][0] > len(subj_index_mapper.special_tokens):
                too_many_unknowns = False
                mention_id_token_ids_map[triple[subj_slot][0]] = triple_segmented[subj_slot]
                if classify_as_too_many_unknowns(triple_segmented[subj_slot]) or triple[subj_slot][0] == UNK:
                    too_many_unknowns = True
                # if triple[rel_slot][0] > len(rel_index_mapper.special_tokens):
                relation_id_token_ids_map[triple[rel_slot][0]] = triple_segmented[rel_slot]
                if classify_as_too_many_unknowns(triple_segmented[rel_slot]) or triple[rel_slot][0] == UNK:
                    too_many_unknowns = True
                # if triple[obj_slot][0] > len(obj_index_mapper.special_tokens):
                mention_id_token_ids_map[triple[obj_slot][0]] = triple_segmented[obj_slot]
                if classify_as_too_many_unknowns(triple_segmented[obj_slot]) or triple[obj_slot][0] == UNK:
                    too_many_unknowns = True
                if not too_many_unknowns:
                    result.append(triple)
            return result

        train_converted_filtered = collect_mentions_and_filter(train_converted)
        others_train_converted_and_filtered = [collect_mentions_and_filter(data) for data in others_train_converted]
        valid_and_test_converted_and_filtered = [collect_mentions_and_filter(data) for data in valid_and_test_converted]

        return train_converted_filtered, others_train_converted_and_filtered + valid_and_test_converted_and_filtered,\
               mention_id_token_ids_map,\
               relation_id_token_ids_map

    # else:
    #     return filter(filter_func, map(itemgetter(0), train_converted)),\
    #            [filter(filter_func, map(itemgetter(0), valid_and_test_converted))
    #             for valid_and_test_converted in others_train_converted + valid_and_test_converted
    #             ],




train = """B O#works in#N Y#E:BO#E:NY
O#works in#W#E:BO#E:W
G B#is president of#USA#E:GB#E:USA
A M#is chancelor of#DE#E:AM#E:DE
G H#is chancelor of#DE#E:GH#E:DE
M#is president of#F#E:M#E:F
A M#works in#Berlin#E:AM#E:B
G H#works in#Bonn#E:GH#E:BN
Bonn#is in#DE#E:BN#E:DE
Berlin#is in#DE#E:B#E:DE
N Y#is in#USA#E:NY#E:USA
W#is in#USA#E:W#E:USA"""

valid = """B O#works in#UNK#E:BO#E:USA
M#is chancelor of#DE#E:AM#E:DE
G H#works in#DE#E:GH#E:NY"""

test = """O#works in#USA#E:BO#E:USA
G B#works in#USA#E:GB#E:USA
A M#works in#DE#E:AM#E:DE"""


train_m = """B O#works in#N Y#B O||| O#N Y||| N Y C
O#works in#W#B O|||O#W
G B#is president of#USA#GB|||B#USA|||US
A M#is chancelor of#DE#A M|||M#DE|||GE
G H#is chancelor of#DE#G H#DE|||GE
M#is president of#F#M#F
A M#works in#Berlin#A M|||M#B
G H#works in#Bonn#G H#BN
Bonn#is in#DE#BN#DE|||GE
Berlin#is in#DE#B#DE|||GE
N Y#is in#USA#N Y|||N Y C#USA|||US
W#is in#USA#W#USA|||US"""

train_smaller_m = """B O#works in#N Y#B O||| O#N Y||| N Y C
N Y#is in#USA#N Y|||N Y C#USA|||US
W#is in#USA#W#USA|||US"""

valid_m = """B O#works in#UNK_M#B O|||O#USA|||US
M#is chancelor of#DE#A M|||M#DE|||GE
G H#works in#DE#E:GH#N Y|||N Y C"""

test_m = """G O#works in#USA#B O|||O#USA|||US
G B#works now in#USA#GB|||B#USA|||US
A M#works at#DE#A M|||M#DE|||GE"""


if __name__ == "__main__":

    # mention_index_mapper = IndexMapper(segment=True)
    # relation_index_mapper = IndexMapper(segment=True)
    #
    # tr|||vl, te, eis_map, ris_map, me_map = convert_datasets_with_entity_annotations(
    #     train=train.split('\n'),
    #     valid=valid.split('\n'),
    #     test=test.split('\n'),
    #     subj_index_mapper=mention_index_mapper,
    #     obj_index_mapper=mention_index_mapper,
    #     rel_index_mapper=relation_index_mapper,
    #     subj_entity_index_mapper=entity_index_mapper,
    #     obj_entity_index_mapper=entity_index_mapper,
    #     triple_format_parser=lambda x: x.strip().split('#'),
    #     segment=True,
    # )
    #
    # print(list(tr), list(vl), list(te))
    #
    # print(eis_map)
    # print(ris_map)
    # print(me_map)
    #
    # print(mention_index_mapper.item2idx)
    # print(mention_index_mapper.item2segmentidx)
    #
    # print(entity_index_mapper.item2idx)
    # print(entity_index_mapper.item2segmentidx)
    #
    # print(relation_index_mapper.item2idx)
    # print(relation_index_mapper.item2segmentidx)


    mention_index_mapper_m = IndexMapper(segment=True)
    relation_index_mapper_m = IndexMapper(segment=True)

    tr_sm, vl_te, me_map, ris_map = convert_datasets_with_entity_mention_annotations(
        train=train_smaller_m.split('\n'),
        valid_and_test=[valid_m.split('\n'), test_m.split('\n')],
        others_train=[train_m.split('\n')],
        subj_index_mapper=mention_index_mapper_m,
        obj_index_mapper=mention_index_mapper_m,
        rel_index_mapper=relation_index_mapper_m,
        triple_format_parser=lambda x: x.strip().split('#'),
        segment=True,
        collect_mention_vocab_also_from_others=False,
    )

    tr, vl, te = vl_te
    print("TRAIN")
    for tri in list(tr):
        print(tri)
    print("TRAIN SMALL")
    for tri in list(tr_sm):
        print(tri)
    print("VALID")
    print(list(vl))
    print("TEST")
    print(list(te))

    print("ENTITY MENTIONS")
    print(list(sorted(me_map.items(), key=itemgetter(0))))
    print("RELATION MENTIONS")
    print(list(sorted(ris_map.items(), key=itemgetter(0))))

    print(len(me_map))
    print(max([k for k,v in me_map.items()]))
    print(max([v for k,v in mention_index_mapper_m.item2idx.items()]))

    print(len(ris_map))
    print(max([k for k,v in ris_map.items()]))
    print(max([v for k,v in relation_index_mapper_m.item2idx.items()]))

    print(mention_index_mapper_m.item2idx)
    print(mention_index_mapper_m.item2segmentidx)

    print(relation_index_mapper_m.item2idx)
    print(relation_index_mapper_m.item2segmentidx)

