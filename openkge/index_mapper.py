# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import codecs
import logging
from collections import Counter, defaultdict
import torch

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '<\s>'

PAD, UNK, BOS, EOS = [0, 1, 2, 3]

class IndexMapper(object):

    special_tokens = {PAD_TOKEN:PAD, UNK_TOKEN:UNK}
    special_tokens_segment = {PAD_TOKEN:PAD, UNK_TOKEN:UNK, BOS_TOKEN:BOS, EOS_TOKEN:EOS}

    def __init__(self,
                 vocab_file=None,
                 threshold=-1,
                 segment_func=lambda line: line.lower().strip().split(),
                 segment_infix='_token',
                 suffix='.txt',
                 additional_tokens=None,
                 segment=True,
                 insert_start=BOS,
                 insert_end=EOS,
                 ):
        self.threshold = threshold
        if additional_tokens is not None:
            self.special_tokens += additional_tokens
        self.item2idx = None
        self.item2segmentidx = None
        self.token_embedding = None
        if vocab_file and os.path.isfile(vocab_file):
                self.load_vocab(vocab_file)
        self.insert_start = insert_start
        self.insert_end = insert_end
        self.segment_func = segment_func
        self.vocab = None
        self.segment_vocab = None
        self.segment_infix = segment_infix
        self.file_type_suffix = suffix
        self.segment = segment
        self._collect_vocab = None
        self._collect_segment_vocab = None

    def set_insert_start_and_end(self,
                                 insert_start=None,
                                 insert_end=None,
                                 ):
        self.insert_start=insert_start
        self.insert_end=insert_end

    def update_item2idx(self):
        item2idx = {item[0]: idx + len(self.special_tokens) for idx, item in enumerate(self.vocab)}
        for tok,i in self.special_tokens.items():
            item2idx[tok] = i
        self.item2idx = defaultdict(lambda: UNK, item2idx)
        if self.segment:
            item2segmentidx = {item[0]: idx + len(self.special_tokens_segment) for idx, item in enumerate(self.segment_vocab)}
            for tok, i in self.special_tokens_segment.items():
                item2segmentidx[tok] = i
            self.item2segmentidx = defaultdict(lambda: UNK, item2segmentidx)

    def get_vocab(self, items, append=True,):
        self.init_vocab(append)
        for item in items:
            self.collect_vocab(item)
        self.finalize_vocab()

    def init_vocab(self, append=True):
        if self._collect_vocab is None or not append:
            self._collect_vocab = Counter()
            self._collect_segment_vocab = Counter()

    def collect_vocab(self, item, segment=True):
        self._collect_vocab[item] += 1
        if self.segment and segment:
            for segm in self.segment_func(item):
                self._collect_segment_vocab[segm] += 1

    def finalize_vocab(self):
        if self._collect_vocab is not None:
            self.vocab = [i for i in self._collect_vocab.most_common() if i[1] > self.threshold]
            if self.segment:
                self.segment_vocab = [i for i in self._collect_segment_vocab.most_common() if i[1] > self.threshold]
            self.update_item2idx()
            self._collect_vocab = None
            self._collect_segment_vocab = None

    def save_vocab(self, vocab_filename, map_suffix='_id_map'):
        if self.vocab is not None:
            with codecs.open(vocab_filename + map_suffix + self.file_type_suffix, 'w', encoding='UTF-8') as f:
                f.write("# token\tid\tcount\t\n")
                for id, key_freq in enumerate(self.vocab):
                    key, freq = key_freq
                    # print(key, id , len(self.special_tokens), freq)
                    f.write("{0}\t{1}\t{2}\n".format(key, id + len(self.special_tokens), freq))
            if self.segment:
                with codecs.open(vocab_filename + self.segment_infix + map_suffix + self.file_type_suffix, 'w', encoding='UTF-8') as f:
                    f.write("# token\tid\tcount\t\n")
                    for id, key_freq in enumerate(self.segment_vocab):
                        key, freq = key_freq
                        f.write("{0}\t{1}\t{2}\n".format(key, id + len(self.special_tokens_segment), freq))

    def load_vocab(self, vocab_filename, limit=None, map_suffix='_id_map'):
        vocab = Counter(self.vocab)
        logging.info("Loading vocab from {}".format(vocab_filename + self.file_type_suffix))
        with codecs.open(vocab_filename + map_suffix + self.file_type_suffix, encoding='UTF-8') as f:
            for line in f:
                if line.startswith('#'): continue
                item, id, count = line.split('\t')
                vocab[item] = int(count)
        self.vocab = vocab.most_common(limit)
        if self.segment:
            segment_vocab = Counter(self.segment_vocab)
            with codecs.open(vocab_filename + self.segment_infix + map_suffix + self.file_type_suffix, encoding='UTF-8') as f:
                for line in f:
                    if line.startswith('#'): continue
                    item, id, count = line.split('\t')
                    segment_vocab[item] = int(count)
            self.segment_vocab = segment_vocab.most_common(limit)
        self.update_item2idx()

    def toidx(self, item, return_tensor=False, insert_start=None, insert_end=None, segment=True):
        if self.segment and segment:
            segmented_item = self.segment_func(item)

        item = [item.strip()]
        mapped_item = []
        mapped_item.extend(map(self.item2idx.__getitem__, item))

        if self.segment and segment:
            insert_start = insert_start if insert_start else self.insert_start
            insert_end = insert_end if insert_end else self.insert_end
            mapped_segmented_item = []
            if insert_start is not None:
                mapped_segmented_item += [insert_start]
            mapped_segmented_item.extend(map(self.item2segmentidx.__getitem__, segmented_item))
            if insert_end is not None:
                mapped_segmented_item += [insert_end]
            if return_tensor:
                return torch.IntTensor(mapped_item), torch.IntTensor(mapped_segmented_item),
            else:
                return mapped_item, mapped_segmented_item
        else:
            if return_tensor:
                return torch.IntTensor(mapped_item), torch.IntTensor(mapped_item)
            else:
                return mapped_item, mapped_item

    def detokenize(self, inputs, delimiter=u' '):
        return delimiter.join([self.idx2item(idx) for idx in inputs]).encode('utf-8')


class IndexMappers:
    SegmentIndexMapper = IndexMapper