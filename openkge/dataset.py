# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import codecs
import itertools
import json
import logging
import os
import pickle
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from os import path

import numpy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from openkge.index_mapper import PAD, EOS, BOS, UNK, IndexMapper
from utils.misc import pack_list_of_lists, unpack_list_of_lists
from utils.metrics import MetricResult


@dataclass
class EntityRelationDatasetMeta:
    entity_id_count_map: dict
    relation_id_count_map: dict
    entity_token_id_count_map: dict
    relation_token_id_count_map: dict
    entity_id_to_tokens_map: dict
    relation_id_to_tokens_map: dict
    entities_size: int
    relations_size: int
    min_entities_size: int
    min_relations_size: int
    entity_tokens_size: int
    relation_tokens_size: int
    max_length: int


class EntityRelationDatasetBase(object):
    def __init__(
        self,
        dataset_dir,
        input_file,
        is_training_data,
        input_style,
        entity_id_map_file="entity_id_map.txt",
        relation_id_map_file="relation_id_map.txt",
        entity_id_tokens_ids_map_file="entity_id_tokens_ids_map.txt",
        relation_id_tokens_ids_map_file="relation_id_tokens_ids_map.txt",
        entity_token_id_map="entity_token_id_map.txt",
        relation_token_id_map="relation_token_id_map.txt",
        replace_entities_by_tokens=False,
        replace_relations_by_tokens=False,
        insert_start=[BOS],
        insert_end=[EOS],
        loss=None,
        max_lengths_tuple=(10, 10),
        device=None,
        copy_data_to_dev_shm=True,
        batch_size=None,
        map_list_to_shm=False,
    ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.input_style = input_style
        self.input_file_name = input_file
        self.is_training_data = is_training_data
        self.device = device
        self.loss = loss
        self.batch_size_for_backward = None
        self.batch_size = batch_size
        self.insert_start = insert_start
        self.insert_end = insert_end
        self.entity_id_map = OrderedDict()
        self.relation_id_map = OrderedDict()
        self.replace_entities_by_tokens = replace_entities_by_tokens
        self.replace_relations_by_tokens = replace_relations_by_tokens
        self.entity_tokens_vocab = OrderedDict()
        self.relation_tokens_vocab = OrderedDict()
        self.map_list_to_shm = map_list_to_shm

        # Begin META fields :TODO refactor this to use EntityRelationDatasetMeta directly

        self.entity_id_map_file = entity_id_map_file
        self.relation_id_map_file = relation_id_map_file
        self.entity_id_tokens_ids_map_file = entity_id_tokens_ids_map_file
        self.relation_id_tokens_ids_map_file = relation_id_tokens_ids_map_file
        self.entity_token_id_map = entity_token_id_map
        self.relation_token_id_map = relation_token_id_map

        self.entity_id_count_map = OrderedDict()
        self.relation_id_count_map = OrderedDict()
        self.entity_token_id_count_map = OrderedDict()
        self.relation_token_id_count_map = OrderedDict()
        self.entity_vocab_size = -1
        self.relations_size = -1
        self.entity_special_vocab_size = max(PAD, UNK) + 1
        self.min_relations_size = max(PAD, UNK) + 1
        self.entity_tokens_size = -1
        self.relation_tokens_size = -1
        self.entity_id_to_tokens_map = None
        self.relation_id_to_tokens_map = None
        self.max_length = max_lengths_tuple

        self.input_file_full_path = os.path.join(dataset_dir, input_file)

        if copy_data_to_dev_shm:
            dev_shm_dataset_dir = "/dev/shm/" + dataset_dir
            dev_shm_input_file = os.path.join(dev_shm_dataset_dir, input_file)
            if not os.path.exists(dev_shm_input_file):
                os.makedirs(dev_shm_dataset_dir, exist_ok=True)
                shutil.copyfile(self.input_file_full_path, dev_shm_input_file)
            self.input_file_full_path = dev_shm_input_file

        self._collect_seen_triples(
            dataset_dir=dataset_dir, input_file=input_file,
        )
        self.load_vocab()

    def get_dataset_meta_dict(self):
        return EntityRelationDatasetMeta(
            entity_id_count_map=self.entity_id_count_map,
            relation_id_count_map=self.relation_id_count_map,
            entity_id_to_tokens_map=self.entity_id_to_tokens_map,
            relation_id_to_tokens_map=self.relation_id_to_tokens_map,
            entity_token_id_count_map=self.entity_token_id_count_map,
            relation_token_id_count_map=self.relation_token_id_count_map,
            entities_size=self.entity_vocab_size,
            relations_size=self.relations_size,
            min_entities_size=self.entity_special_vocab_size,
            min_relations_size=self.min_relations_size,
            entity_tokens_size=self.entity_tokens_size,
            relation_tokens_size=self.relation_tokens_size,
            max_length=self.max_length,
        )

    def load_vocab(self):

        if path.exists(os.path.join(self.dataset_dir, "{}-entity_relation_maps.pickle".format(self.input_file_name))):

            logging.info(
                "loading cached data from {}".format(
                    os.path.join(self.dataset_dir, "{}-entity_relation_maps.pickle".format(self.input_file_name))
                )
            )
            with open(
                os.path.join(self.dataset_dir, "{}-entity_relation_maps.pickle".format(self.input_file_name)), "rb"
            ) as f:
                (
                    self.entity_id_map,
                    self.entity_id_count_map,
                    self.entity_vocab_size,
                    self.entity_tokens_size,
                    self.entity_id_to_tokens_map,
                    self.entity_tokens_vocab,
                    self.entity_token_id_count_map,
                    self.relation_id_map,
                    self.relation_id_count_map,
                    self.relations_size,
                    self.relation_tokens_size,
                    self.relation_id_to_tokens_map,
                    self.relation_tokens_vocab,
                    self.relation_token_id_count_map,
                ) = pickle.load(f)

        else:

            if self.entity_id_map_file:
                with codecs.open(os.path.join(self.dataset_dir, self.entity_id_map_file), encoding="UTF-8") as f:
                    lines = 0
                    for line in f.readlines():
                        lines += 1
                        if line.startswith("#") and lines == 1:
                            continue
                        entity, e_id, count = line.split("\t")
                        e_id = int(e_id)
                        self.entity_id_map[entity] = e_id
                        self.entity_id_count_map[e_id] = int(count)
                        self.entity_vocab_size = max(self.entity_vocab_size, e_id)
                    for st, i in IndexMapper.special_tokens.items():
                        self.entity_id_count_map[i] = 1

            if os.path.exists(os.path.join(self.dataset_dir, self.entity_id_tokens_ids_map_file)):
                with codecs.open(
                    os.path.join(self.dataset_dir, self.entity_id_tokens_ids_map_file), encoding="UTF-8"
                ) as f:
                    entity_id_to_tokens_map = OrderedDict()
                    lines = 0
                    for line in f.readlines():
                        lines += 1
                        if line.startswith("#") and lines == 1:
                            continue
                        e_id, tok_ids = line.strip().split("\t")
                        e_id = int(e_id)
                        tok_ids = list(map(int, tok_ids.split()))
                        entity_id_to_tokens_map[e_id] = tok_ids
                        self.entity_tokens_size = max(self.entity_tokens_size, max(tok_ids))
                    for st, i in IndexMapper.special_tokens.items():
                        entity_id_to_tokens_map[i] = [1]
                    assert (
                        self.entity_vocab_size == len(entity_id_to_tokens_map) - 1
                    ), "self.entity_size {} == len(entities_id_to_tokens_map)-1 {}".format(
                        self.entity_vocab_size, len(entity_id_to_tokens_map) - 1
                    )
                    self.entity_id_to_tokens_map = tuple(
                        [v for k, v in sorted(entity_id_to_tokens_map.items(), key=lambda x: x[0])]
                    )
                if self.entity_token_id_map:
                    with codecs.open(os.path.join(self.dataset_dir, self.entity_token_id_map), encoding="UTF-8") as f:
                        lines = 0
                        for line in f.readlines():
                            lines += 1
                            if line.startswith("#") and lines == 1:
                                continue
                            word, id, count = line.strip().split("\t")
                            id = int(id)
                            self.entity_tokens_vocab[word] = id
                            self.entity_token_id_count_map[id] = int(count)
                        for st, i in IndexMapper.special_tokens_segment.items():
                            self.entity_token_id_count_map[i] = 1

            if self.relation_id_map_file:
                with codecs.open(os.path.join(self.dataset_dir, self.relation_id_map_file), encoding="UTF-8") as f:
                    lines = 0
                    for line in f.readlines():
                        lines += 1
                        if line.startswith("#") and lines == 1:
                            continue
                        relation, r_id, count = line.split("\t")
                        r_id = int(r_id)
                        self.relation_id_map[relation] = r_id
                        self.relation_id_count_map[r_id] = int(count)
                        self.relations_size = max(self.relations_size, r_id)
                    for st, i in IndexMapper.special_tokens.items():
                        self.relation_id_count_map[i] = 1

            if os.path.exists(os.path.join(self.dataset_dir, self.relation_id_tokens_ids_map_file)):
                with codecs.open(
                    os.path.join(self.dataset_dir, self.relation_id_tokens_ids_map_file), encoding="UTF-8"
                ) as f:
                    relation_id_to_tokens_map = OrderedDict()
                    lines = 0
                    for line in f.readlines():
                        lines += 1
                        if line.startswith("#") and lines == 1:
                            continue
                        r_id, tok_ids = line.strip().split("\t")
                        r_id = int(r_id)
                        tok_ids = list(map(int, tok_ids.split()))
                        relation_id_to_tokens_map[r_id] = tok_ids
                        self.relation_tokens_size = max(self.relation_tokens_size, max(tok_ids))
                    for st, i in IndexMapper.special_tokens.items():
                        relation_id_to_tokens_map[i] = [1]
                    assert (
                        self.relations_size == len(relation_id_to_tokens_map) - 1
                    ), "self.relation_size {} == len(relation_id_to_tokens_map)-1 {}".format(
                        self.relations_size, len(relation_id_to_tokens_map) - 1
                    )
                    self.relation_id_to_tokens_map = tuple(
                        [v for k, v in sorted(relation_id_to_tokens_map.items(), key=lambda x: x[0])]
                    )
                if self.relation_token_id_map:
                    with codecs.open(os.path.join(self.dataset_dir, self.relation_token_id_map), encoding="UTF-8") as f:
                        lines = 0
                        for line in f.readlines():
                            lines += 1
                            if line.startswith("#") and lines == 1:
                                continue
                            word, id, count = line.strip().split("\t")
                            id = int(id)
                            self.relation_tokens_vocab[word] = id
                            self.relation_token_id_count_map[id] = int(count)
                        for st, i in IndexMapper.special_tokens_segment.items():
                            self.relation_token_id_count_map[i] = 1

            with open(
                os.path.join(self.dataset_dir, "{}-entity_relation_maps.pickle".format(self.input_file_name)), "wb"
            ) as f:
                pickle.dump(
                    (
                        self.entity_id_map,
                        self.entity_id_count_map,
                        self.entity_vocab_size,
                        self.entity_tokens_size,
                        self.entity_id_to_tokens_map,
                        self.entity_tokens_vocab,
                        self.entity_token_id_count_map,
                        self.relation_id_map,
                        self.relation_id_count_map,
                        self.relations_size,
                        self.relation_tokens_size,
                        self.relation_id_to_tokens_map,
                        self.relation_tokens_vocab,
                        self.relation_token_id_count_map,
                    ),
                    f,
                )

        self.entity_vocab_size += 1
        self.relations_size += 1
        self.entity_tokens_size += 1
        self.relation_tokens_size += 1

        self.vocab_starts_tuple = (4, 4)
        self.vocab_ends_tuple = (self.entity_tokens_size, self.relation_tokens_size)

    def _collect_seen_triples(
        self, dataset_dir, input_file,
    ):
        raise NotImplementedError

    def merge_all_splits_triples(
        self, dataset_dir, train_input_file, valid_input_file, test_input_file,
    ):
        raise NotImplementedError

    def create_data_tensors(
        self, dataset_dir, train_input_file, valid_input_file, test_input_file,
    ):
        raise NotImplementedError

    @staticmethod
    def compute_metrics(
            filter_mask,
            label_ids,
            predictions,

    ) -> MetricResult:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_loader(
        self, shuffle=False, sampler=None, num_workers=4, pin_memory=False, drop_last=True,
    ):
        raise NotImplementedError

    def input_and_labels_to_device(self, data, training, device):
        raise NotImplementedError

class OneToNMentionRelationDataset(EntityRelationDatasetBase):
    def __init__(self,
                 use_batch_shared_entities=False,
                 min_size_batch_labels=-1,
                 max_size_prefix_label=-1,
                 **kwargs):
        super().__init__(
            input_style="right_and_left_prefix",
            **kwargs)
        logging.info(
            """
    Dataset {}
        Meta:
            location: {}
            entity min id - max id: {} - {}
            relation min id - max id: {} - {}
            batch size: {}
            batch shared labels: {}
        Labels:
            loss: {}            
                    """.format(
                self.__class__.__name__,
                self.input_file_full_path,
                self.entity_vocab_size,
                self.entity_special_vocab_size,
                self.relations_size,
                self.min_relations_size,
                self.batch_size,
                use_batch_shared_entities,
                self.loss,
            )
        )
        self.min_size_batch_labels = min_size_batch_labels
        self.max_size_prefix_label = max_size_prefix_label
        self.use_batch_shared_entities = use_batch_shared_entities

    def __len__(self):
        return len(self.seen_prefixes_tensor)

    def input_and_labels_to_device(self, data, training, device):
        (
            slot_input_tensor,
            normalizer_loss,
            normalizer_metric,
            label_tensor,
            label_ids,
            filter_mask_tensor,
            batch_shared_label_ids,
        ) = data
        slot_inputs_new = list()

        for slot_input in slot_input_tensor:
            if slot_input is not None:
                (pref_1, pref_2) = slot_input
                # if device:
                pref_1 = pref_1.to(device)
                pref_2 = pref_2.to(device)
                slot_inputs_new.append((pref_1, pref_2))
            else:
                slot_inputs_new.append(None)

        label_tensor = label_tensor.to(device)
        if batch_shared_label_ids is not None:
            batch_shared_label_ids = batch_shared_label_ids.to(device)
        if not training:
            filter_mask_tensor = filter_mask_tensor.to(device)

        return (
            slot_inputs_new,
            normalizer_loss,
            normalizer_metric,
            label_tensor,
            label_ids,
            filter_mask_tensor,
            batch_shared_label_ids,
        )

    @staticmethod
    def compute_metrics(
            filter_mask,
            label_ids,
            predictions,
    ) -> MetricResult:
        result = MetricResult()
        for prefix_filter, prefix_labels, prefix_prediction in zip(filter_mask, label_ids, predictions):

            prefix_prediction_repeat = prefix_prediction.unsqueeze(0).repeat(len(prefix_labels), 1)
            prefix_filters_repeat = prefix_filter.unsqueeze(0).repeat(len(prefix_labels), 1)
            true_prediction_list = list()

            for prefix_label in prefix_labels:
                true_prediction_list.append(prefix_prediction[prefix_label.long()].max(0)[0])
            true_prediction = torch.Tensor(true_prediction_list).to(prefix_prediction.device)

            prefix_prediction_repeat.masked_fill_(prefix_filters_repeat, -1e8)
            false_positives = (
                (true_prediction.view(len(prefix_labels), -1) < prefix_prediction_repeat).long().sum(1)
            )
            equals = (true_prediction.view(len(prefix_labels), -1) == prefix_prediction_repeat).long().sum(1)
            ranks = false_positives + equals // 2
            divide_by = len(true_prediction_list)
            result["mrr"].update((1.0 / (ranks + 1).float()).sum().item()/divide_by, divide_by)
            result["mr"].update(ranks.sum().item()/divide_by, divide_by)
            result["h50"].update((ranks < 50).float().sum().item()/divide_by, divide_by)
            result["h10"].update((ranks < 10).float().sum().item()/divide_by, divide_by)
            result["h3"].update((ranks < 3).float().sum().item()/divide_by, divide_by)
            result["h1"].update((ranks < 1).float().sum().item()/divide_by, divide_by)
        return result

    def get_loader(
        self, shuffle=False, sampler=None, num_workers=4, pin_memory=False, drop_last=True,
    ):
        def collate(triple):
            return OneToNMentionRelationDataset_collate_func(
                sp_po__batch=triple,
                entity_vocab_size=self.entity_vocab_size,
                entity_vocab_offset=self.entity_special_vocab_size,
                min_size_batch_labels=self.min_size_batch_labels,
                is_training_data=self.is_training_data,
                this_split_entities_list=self.seen_entities_tensor,
                all_splits_entities_tensor=self.all_splits_entities_tensor,
                use_batch_shared_entities=self.use_batch_shared_entities,
            )

        return torch.utils.data.DataLoader(
            self.seen_prefixes_tensor,
            batch_size=self.batch_size,
            collate_fn=collate,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def _collect_seen_triples(self, dataset_dir, input_file):

        if not os.path.exists(f"{dataset_dir}/{self.__class__.__name__}-{input_file}-sp_o.jsonl") or not os.path.exists(
            f"{dataset_dir}/{self.__class__.__name__}-{input_file}-po_s.jsonl"
        ):

            logging.info(f"Creating {dataset_dir}/{self.__class__.__name__}-{input_file}-PREFIX_SLOT.jsonl")

            for prefix_str, pref_p, pref_e, pref_1, pref_2, slot_id, slot_ents_id in [
                ("sp_o", 1, 0, 0, 1, 2, 4),
                ("po_s", 1, 2, 1, 2, 0, 3),
            ]:
                with open(f"{dataset_dir}/{input_file}") as f:
                    with open(f"{dataset_dir}/{self.__class__.__name__}-{input_file}-{prefix_str}.jsonl", "w") as f_out:
                        prefix_entities = None
                        for line in tqdm(
                            sorted(
                                sorted(f.readlines(), key=lambda l: l.split("\t")[pref_p]),
                                key=lambda l: l.split("\t")[pref_e],
                            )
                        ):
                            fields = line.split("\t")
                            prefix = (int(fields[pref_1]), int(fields[pref_2]))
                            if prefix_entities is None:
                                prefix_entities = {
                                    "prefix": prefix,
                                    "entities": [[int(i) for i in fields[slot_ents_id].split()]],
                                    "slot": slot_id,
                                }
                            elif prefix == prefix_entities["prefix"]:
                                prefix_entities["entities"].append([int(i) for i in fields[slot_ents_id].split()])
                            else:
                                f_out.writelines(json.dumps(prefix_entities) + "\n")
                                prefix_entities = {
                                    "prefix": prefix,
                                    "entities": [[int(i) for i in fields[slot_ents_id].split()]],
                                    "slot": slot_id,
                                }

    def merge_all_splits_triples(
        self, dataset_dir, train_input_file, valid_input_file, test_input_file,
    ):

        if not os.path.exists(
            f"{dataset_dir}/{self.__class__.__name__}-{train_input_file}-{valid_input_file}-{test_input_file}-sp_o.jsonl"
        ) or not os.path.exists(
            f"{dataset_dir}/{self.__class__.__name__}-{train_input_file}-{valid_input_file}-{test_input_file}-po_s.jsonl"
        ):

            logging.info(
                f"Creating {dataset_dir}/{self.__class__.__name__}-{train_input_file}-{valid_input_file}-{test_input_file}-PREFIX_SLOT.jsonl"
            )

            for prefix_str in ["sp_o", "po_s"]:
                with open(f"{dataset_dir}/{self.__class__.__name__}-{train_input_file}-{prefix_str}.jsonl") as ftrain:
                    with open(
                        f"{dataset_dir}/{self.__class__.__name__}-{valid_input_file}-{prefix_str}.jsonl"
                    ) as fvalid:
                        with open(
                            f"{dataset_dir}/{self.__class__.__name__}-{test_input_file}-{prefix_str}.jsonl"
                        ) as ftest:
                            with open(
                                f"{dataset_dir}/{self.__class__.__name__}-{train_input_file}-{valid_input_file}-{test_input_file}-{prefix_str}.jsonl",
                                "w",
                            ) as fout:
                                previous = None
                                for line in tqdm(
                                    sorted(
                                        ftrain.readlines() + fvalid.readlines() + ftest.readlines(),
                                        key=lambda l: json.loads(l)["prefix"],
                                    )
                                ):
                                    prefix_entities = json.loads(line)
                                    if previous is None:
                                        previous = prefix_entities
                                        previous["entities"] = set(itertools.chain(*previous["entities"]))
                                    elif previous["prefix"] == prefix_entities["prefix"]:
                                        previous["entities"].update(itertools.chain(*prefix_entities["entities"]))
                                    else:
                                        previous["entities"] = list(previous["entities"])
                                        fout.writelines(json.dumps(previous) + "\n")
                                        previous = prefix_entities
                                        previous["entities"] = set(itertools.chain(*previous["entities"]))
                                previous["entities"] = list(previous["entities"])
                                fout.writelines(json.dumps(previous) + "\n")

    def create_data_tensors(
        self, dataset_dir, train_input_file, valid_input_file, test_input_file,
    ):

        all_splits_coords, self.all_splits_entities_tensor = None, None

        all_tensor_file = f"{dataset_dir}/{self.__class__.__name__}-{train_input_file}-{valid_input_file}-{test_input_file}-all_entities.pickle"
        if os.path.exists(all_tensor_file):
            with open(all_tensor_file, "rb") as f:
                logging.info(f"Loading {all_tensor_file}")
                all_splits_coords, self.all_splits_entities_tensor = pickle.load(f)
        else:

            logging.info(f"Creating {all_tensor_file}")

            all_splits_coords = {
                "sp_o": dict(),
                "po_s": dict(),
            }
            all_splits_entities_size = 0
            for prefix_str in ["sp_o", "po_s"]:
                with open(
                    f"{dataset_dir}/{self.__class__.__name__}-{train_input_file}-{valid_input_file}-{test_input_file}-{prefix_str}.jsonl"
                ) as f:
                    for line in tqdm(f.readlines()):
                        all_splits_entities_size += int(len(json.loads(line)["entities"]))

            all_splits_entities_list_offset = 0
            self.all_splits_entities_tensor = torch.IntTensor(all_splits_entities_size)
            for prefix_str in ["sp_o", "po_s"]:
                with open(
                    f"{dataset_dir}/{self.__class__.__name__}-{train_input_file}-{valid_input_file}-{test_input_file}-{prefix_str}.jsonl"
                ) as f:
                    for line in tqdm(f.readlines()):
                        prefix_entities = json.loads(line)
                        start, end = (
                            all_splits_entities_list_offset,
                            all_splits_entities_list_offset + len(prefix_entities["entities"]),
                        )
                        self.all_splits_entities_tensor[start:end] = torch.IntTensor(prefix_entities["entities"])
                        all_splits_coords[prefix_str][tuple(prefix_entities["prefix"])] = (start, end)
                        all_splits_entities_list_offset += len(prefix_entities["entities"])

            with open(all_tensor_file, "wb") as f:
                pickle.dump((all_splits_coords, self.all_splits_entities_tensor), f)

        # Create prefix and seen entities tensor

        input_tensor_file = f"{dataset_dir}/{self.__class__.__name__}-{self.input_file_name}-{self.max_size_prefix_label}-tensor.pickle"

        if os.path.exists(input_tensor_file):
            with open(input_tensor_file, "rb") as f:
                logging.info(f"Loading {input_tensor_file}")
                self.seen_entities_tensor, self.seen_prefixes_tensor = pickle.load(f)
        else:

            logging.info(f"Creating {input_tensor_file}")

            seen_prefixes_size = 0
            seen_entities_size = 0
            for prefix_str in ["sp_o", "po_s"]:
                with open(f"{dataset_dir}/{self.__class__.__name__}-{self.input_file_name}-{prefix_str}.jsonl") as fin:
                    for line in tqdm(fin.readlines()):
                        entities = json.loads(line)["entities"]
                        if (
                            self.is_training_data
                            and self.max_size_prefix_label > 1
                            and len(entities) > self.max_size_prefix_label
                        ):
                            for offset in range(
                                0,
                                len(entities) + self.max_size_prefix_label,
                                self.max_size_prefix_label,
                            ):
                                _entities = pack_list_of_lists(
                                    entities[offset : offset + self.max_size_prefix_label]
                                )
                                seen_prefixes_size += 1
                                seen_entities_size += len(_entities)
                        else:
                            entities = pack_list_of_lists(entities)
                            seen_prefixes_size += 1
                            seen_entities_size += len(entities)

            self.seen_prefixes_tensor = torch.IntTensor(seen_prefixes_size, 7)
            self.seen_entities_tensor = torch.IntTensor(seen_entities_size)

            seen_prefixes_offset = 0
            seen_entities_list_offset = 0

            for prefix_str in ["sp_o", "po_s"]:
                with open(f"{dataset_dir}/{self.__class__.__name__}-{self.input_file_name}-{prefix_str}.jsonl") as fin:
                    for line in tqdm(fin.readlines()):

                        prefix_entities = json.loads(line)
                        prefix, entities, slot = (
                            prefix_entities["prefix"],
                            prefix_entities["entities"],
                            prefix_entities["slot"],
                        )

                        if (
                            self.is_training_data
                            and self.max_size_prefix_label > 1
                            and len(entities) > self.max_size_prefix_label
                        ):
                            for offset in range(0, len(entities), self.max_size_prefix_label):
                                _entities = pack_list_of_lists(
                                    entities[offset : offset + self.max_size_prefix_label]
                                )
                                self.seen_prefixes_tensor[seen_prefixes_offset] = torch.IntTensor(
                                    [
                                        *prefix,
                                        seen_entities_list_offset,
                                        seen_entities_list_offset + len(_entities),
                                        all_splits_coords[prefix_str][tuple(prefix)][0] if not self.is_training_data else 0,
                                        all_splits_coords[prefix_str][tuple(prefix)][1] if not self.is_training_data else 0,
                                        int(slot),
                                    ]
                                )
                                start, end = seen_entities_list_offset, seen_entities_list_offset + len(_entities)
                                self.seen_entities_tensor[start:end] = torch.IntTensor(_entities)
                                seen_prefixes_offset += 1
                                seen_entities_list_offset += len(_entities)

                        else:
                            entities = pack_list_of_lists(entities)
                            self.seen_prefixes_tensor[seen_prefixes_offset] = torch.IntTensor(
                                [
                                    *prefix,
                                    seen_entities_list_offset,
                                    seen_entities_list_offset + len(entities),
                                    all_splits_coords[prefix_str][tuple(prefix)][0] if not self.is_training_data else 0,
                                    all_splits_coords[prefix_str][tuple(prefix)][1] if not self.is_training_data else 0,
                                    int(slot),
                                ]
                            )
                            start, end = seen_entities_list_offset, seen_entities_list_offset + len(entities)
                            self.seen_entities_tensor[start:end] = torch.IntTensor(entities)
                            seen_prefixes_offset += 1
                            seen_entities_list_offset += len(entities)

            with open(input_tensor_file, "wb") as f:
                pickle.dump((self.seen_entities_tensor, self.seen_prefixes_tensor), f)


# Uncomment and replace the following for profiling

# def OneToNEntityRelationDataset_collate_func(
#     *args, **kwargs,
# ):
#     import cProfile
#     result = [None]
#     cProfile.runctx('result[0] = OneToNEntityRelationDataset_collate_func(*args, **kwargs,)', globals(), locals(), '/home/sbrosche/PycharmProjects/profiling/prof%d.prof' % torch.multiprocessing.current_process().pid)
#     return result[0]
#
# def OneToNMentionRelationDataset_collate_func(
def OneToNMentionRelationDataset_collate_func(
        use_batch_shared_entities,
        sp_po__batch,
        entity_vocab_size,
        entity_vocab_offset,
        is_training_data,
        this_split_entities_list,
        all_splits_entities_tensor,
        min_size_batch_labels=0,
):
    slot_batch_sizes = dict()
    slot_items = dict()
    for slot in [0, 2]:
        slot_batch_sizes[slot] = 0
        slot_items[slot] = list()

    batch_entity_ids_to_label_id_dict = OrderedDict()

    # In the following the batch is constructed in generally two ways:
    #
    # Either use_batch_shared_entities is True, then the labels
    # (== entity id) for each sp or po prefix are translated into with a
    # mapping from entity vocabulary ids to the ids of the set of the entity
    # ids within the batch.
    #
    # For example,
    #
    # prefixes_in_batch =
    #
    # [
    #   (1, 2)   <-- slot == 0 means this is a tuple of (relation_id, object entity_id)
    #   (3, 1)   <-- slot == 1 means this is a tuple of (subject entity_id, relation_id)
    # ]
    #
    # entities_in_batch = [
    #   [0, 2]   <- slot == 0 means this are all subject entity ids
    #   [2, 5000]   <- slot == 1 means this are all object entity ids
    # ]
    #
    # Then the mapping collect from all entity answers form the batch is
    #
    # -> batch_seen_entity_ids_to_label_id_dict {
    #   0 : 0
    #   2 : 1
    #   5000 : 2
    # }
    #
    # Thus we can map the labels (== entities_in_batch) to
    #
    # labels = [
    #   [0, 1]     <- mapped from [0, 2]
    #   [1, 2]  <- mapped from [2, 5000]
    # ]
    #
    # Or the entities and labels are *all* the entities in the entity vocab
    #


    #
    # Collect batch items and group them by slot, i.e. by sp_o and po_s examples
    #

    for batch_item_id, sp_po in enumerate(sp_po__batch):
        (
            sp_po_0,
            sp_po_1,
            this_split_start,
            this_split_end,
            all_splits_start,
            all_splits_end,
            slot,
        ) = sp_po.tolist()

        this_split_entities_in_batch, this_split_entities_in_batch_unpacked = \
            unpack_list_of_lists(this_split_entities_list[this_split_start:this_split_end])
        all_splits_entities_in_batch = all_splits_entities_tensor[all_splits_start:all_splits_end].tolist()

        slot_batch_sizes[slot] += 1

        sp_po_tuple = (sp_po_0, sp_po_1)
        slot_items[slot].append(
            (
                sp_po_tuple,
                this_split_entities_in_batch,
                this_split_entities_in_batch_unpacked,
                all_splits_entities_in_batch
            )
        )

        if use_batch_shared_entities:
            if is_training_data:
                # collect batch entity ids only from this split
                for ent_id in this_split_entities_in_batch_unpacked:
                    batch_entity_ids_to_label_id_dict[ent_id] = batch_entity_ids_to_label_id_dict.get(
                        ent_id, len(batch_entity_ids_to_label_id_dict)
                    )
            else:
                # collect batch entity ids from all splits ( == train, valid and test)
                for ent_id in all_splits_entities_in_batch:
                    batch_entity_ids_to_label_id_dict[ent_id] = batch_entity_ids_to_label_id_dict.get(
                        ent_id, len(batch_entity_ids_to_label_id_dict)
                    )

    sp_and_po_batch_size = slot_batch_sizes[0] + slot_batch_sizes[2]

    #
    # Determine label size if use_batch_shared_entities and create the batch label ids
    #

    if use_batch_shared_entities > 0:

        if min_size_batch_labels is None or min_size_batch_labels < 0:
            min_size_batch_labels = 0

        # To create a label tensor we have to know the size of labels and if the
        # number of batch_shared_labels is smaller than min_size_batch_labels, then
        # we fill it up with random sampled entities not seen in the batch or
        # adjust the min_size_batch_labels.

        if len(batch_entity_ids_to_label_id_dict.keys()) >= min_size_batch_labels:

            logging.debug("Size of unique entity ids in batch is larger than min_size_batch_labels. "
                         "Either increase min_size_batch_labels or decrease max_size_prefix_label.")

            batch_shared_entity_ids = list(batch_entity_ids_to_label_id_dict.keys())
            min_size_batch_labels = len(batch_shared_entity_ids)

        else:

            negative_samples = set(
                numpy.random.choice(entity_vocab_size - entity_vocab_offset, min_size_batch_labels, replace=False) + entity_vocab_offset
            )
            negative_samples.difference_update(batch_entity_ids_to_label_id_dict.keys())
            batch_shared_entity_ids = (
                                             list(batch_entity_ids_to_label_id_dict.keys()) +
                                             list(negative_samples)
                                     )[:min_size_batch_labels]


        batch_shared_entity_ids = torch.IntTensor(batch_shared_entity_ids).unsqueeze(1)
        label_tensor = torch.zeros((sp_and_po_batch_size, min_size_batch_labels))

        if not is_training_data:
            filter_mask_tensor = torch.zeros((sp_and_po_batch_size, min_size_batch_labels)).bool()
            label_ids = list()

    else:

        batch_shared_entity_ids = torch.arange(entity_vocab_size)[entity_vocab_offset:].int().unsqueeze(1)
        label_tensor = torch.zeros((sp_and_po_batch_size, entity_vocab_size - entity_vocab_offset))
        if not is_training_data:
            filter_mask_tensor = torch.zeros((sp_and_po_batch_size, entity_vocab_size - entity_vocab_offset)).bool()
            label_ids = list()

    #
    # Now construct the batch tensors for sp and po prefixes, labels and (if evaluating) then also filter masks
    #

    sp_po_input_tensors = list()
    batch_offset = 0

    for slot in [0, 2]:
        # If no examples for sp or po have been collected, then continue
        if slot_batch_sizes[slot] == 0:
            sp_po_input_tensors.append(None)
            continue

        sp_po_batch_tensor = torch.zeros((slot_batch_sizes[slot], 2)).int()
        slot_offset = 0

        for sp_po_tuple, \
            this_split_entities_in_batch, \
            this_split_entities_in_batch_unpacked, \
            all_splits_entities_in_batch in slot_items[slot]:

            sp_po_batch_tensor.narrow(0, slot_offset, 1).copy_(torch.IntTensor(sp_po_tuple))

            if use_batch_shared_entities:

                label_tensor[batch_offset][
                    torch.LongTensor(list(map(batch_entity_ids_to_label_id_dict.__getitem__, this_split_entities_in_batch_unpacked)))
                ] = 1
                if not is_training_data:
                    label_ids_this_prefix = list()
                    for this_split_entity_idx, this_split_entity in enumerate(this_split_entities_in_batch):
                        label_ids_this_prefix.append(
                            torch.IntTensor(
                                list(map(batch_entity_ids_to_label_id_dict.__getitem__, this_split_entity))
                            )
                        )
                    label_ids.append(label_ids_this_prefix)
                    filter_mask_tensor[batch_offset][
                        torch.LongTensor(list(map(batch_entity_ids_to_label_id_dict.__getitem__, all_splits_entities_in_batch)))
                    ] = 1

            else:

                label_tensor[batch_offset][torch.LongTensor(this_split_entities_in_batch_unpacked) - entity_vocab_offset] = 1
                if not is_training_data:
                    label_ids_this_prefix = list()
                    for this_split_entity_idx, this_split_entity in enumerate(this_split_entities_in_batch):
                        label_ids_this_prefix.append(torch.IntTensor(this_split_entity) - entity_vocab_offset)
                    label_ids.append(label_ids_this_prefix)
                    filter_mask_tensor[batch_offset][torch.LongTensor(all_splits_entities_in_batch) - entity_vocab_offset] = 1

            slot_offset += 1
            batch_offset += 1

        sp_po_input_tensors.append(sp_po_batch_tensor.chunk(2, dim=1))

    normalizer_metric = label_tensor.sum().item()
    normalizer_loss = label_tensor.size(0)*label_tensor.size(1)

    if is_training_data:
        return sp_po_input_tensors, normalizer_loss, normalizer_metric, label_tensor, None, None, batch_shared_entity_ids
    else:
        return sp_po_input_tensors, normalizer_loss, normalizer_metric, label_tensor, label_ids, filter_mask_tensor, batch_shared_entity_ids


class Datasets:
    OneToNMentionRelationDataset = OneToNMentionRelationDataset
