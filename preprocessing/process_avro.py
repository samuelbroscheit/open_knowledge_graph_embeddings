import glob
import pickle
import multiprocessing
import time
from collections import Counter
from typing import Dict

from avro.datafile import DataFileReader
from avro.io import DatumReader

from preprocessing.pipeline_job import PipelineJob

from preprocessing.misc import normalize_wiki_entity, prettyformat_dict_string


def process_triple(j, AVRO_FILE, triple, start_time, len_local_entity_mentions_map, job_object : PipelineJob):

    if triple['confidence_score'] < 0.3:
        return None

    if 'PRP$' in [w['pos'] for w in triple['dropped_words_subject']]:
        return None

    if 'no' in [v for k, v in triple['quantities'].items()]:
        return None

    sentence = [w['word'] for w in triple['sentence_linked']['tokens']]

    if j % job_object.opts.process_avro__log_every == 0 and j > 0:
        job_object.log('{} {} triples processed {} entities collected {} sec'.format(AVRO_FILE, j, len_local_entity_mentions_map, (time.time() - start_time)))
        # start_time = time.time()
        # break

    subject_word = [w['word'] if 'QUANT' not in w['word'] else triple['quantities'][w['word'][6:]] if w['word'][6:] in triple['quantities'] else w['word'] for w in sorted(triple['subject'] + triple['dropped_words_subject'], key=lambda x: x['index'])]
    subject_word_lc = [w.lower() for w in subject_word]
    relation_word = [w['word'] if 'QUANT' not in w['word'] else triple['quantities'][w['word'][6:]] if w['word'][6:] in triple['quantities'] else w['word'] for w in sorted(triple['relation'] + triple['dropped_words_relation'], key=lambda x: x['index'])]
    relation_word_lc = [w.lower() for w in relation_word]
    object_word = [w['word'] if 'QUANT' not in w['word'] else triple['quantities'][w['word'][6:]] if w['word'][6:] in triple['quantities'] else w['word'] for w in sorted(triple['object'] + triple['dropped_words_object'], key=lambda x: x['index'])]
    object_word_lc = [w.lower() for w in object_word]

    if relation_word == ['is:impl_appos-clause']:
        return None

    if len([w['word'] for w in triple['subject']]) == 0:
        if job_object.opts.verbose: print('len([w["word"] for w in triple["subject"]]) == 0', '\n\n--------------------------')
        return None
    if len([w['word'] for w in triple['object']]) == 0:
        if job_object.opts.verbose: print('len([w["word"] for w in triple["object"]]) == 0', '\n\n--------------------------')
        return None
    if [w['pos'] for w in triple['subject']][-1] in ['RB', 'WDT']:
        return None
    if [w['pos'] for w in triple['subject']][-1] in ['DT', 'PRP', 'PRP$'] and triple['subject'][-1]['word'] not in ['I']:
        if job_object.opts.verbose: print("[w['pos'] for w in triple['subject']][-1] in ['DT', 'PRP', 'PRP$']")
        if job_object.opts.verbose: print('\n\n', sentence, '\n\n', triple['quantities'], '\n\n', subject_word, relation_word, object_word, prettyformat_dict_string(triple), '\n\n--------------------------')
        return None
    if [w['pos'] for w in triple['object']][-1] in ['RB', 'WDT']:
        return None
    if [w['pos'] for w in triple['object']][-1] in ['DT', 'PRP', 'PRP$'] and triple['object'][-1]['word'] not in ['I']:
        if job_object.opts.verbose: print("w['pos'] for w in triple['object']][-1] in ['DT', 'PRP', 'PRP$']")
        if job_object.opts.verbose: print('\n\n', sentence, '\n\n', triple['quantities'], '\n\n', subject_word, relation_word, object_word, prettyformat_dict_string(triple), '\n\n--------------------------')
        return None

    # if len([w['word'] for w in triple['subject'] if 'QUANT' in w['word']]) > 0 : print("'QUANT' in subject", '\n\n', sentence, '\n\n', triple['quantities'], '\n\n', subject_word, relation_word, object_word, prettyformat_dict_string(triple), '\n\n--------------------------')
    # if len([w['word'] for w in triple['object'] if 'QUANT' in w['word']]) > 0 : print("'QUANT' in object", '\n\n', sentence, '\n\n', triple['quantities'], '\n\n', subject_word, relation_word, object_word, prettyformat_dict_string(triple), '\n\n--------------------------')

    if \
            len(subject_word) == 0 or \
            len(object_word) == 0 \
            :
        if job_object.opts.verbose: print("len(_bject_word) == 0", '\n\n--------------------------')
        return None

    if \
            len(subject_word) > job_object.opts.process_avro__len_subject_word or \
            len(object_word) > job_object.opts.process_avro__len_object_word \
            :
        if job_object.opts.verbose: print("len(_bject_word) > 10 ", '\n\n--------------------------')
        return None

    return subject_word, relation_word, object_word, subject_word_lc, relation_word_lc, object_word_lc


class Worker(multiprocessing.Process):

    def __init__(self, job, in_queue, out_queue):
        super().__init__()
        self.job = job
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        # this loop will run until it receives None form the in_queue, if the queue is empty
        #  the loop will wait until it gets something
        for next_item in iter(self.in_queue.get, None):
            AVRO_FILE = next_item
            self.out_queue.put((self.extract_data(next_item), AVRO_FILE))

    def extract_data(self, AVRO_FILE):

        reader = DataFileReader(open(AVRO_FILE, "rb"), DatumReader())

        start_time = time.time()
        local_entity_mentions_map = dict()
        local_entity_mentions_map_lc = dict()
        local_triples_list = list()
        local_relation_counter = Counter()
        local_triples_list_lc = list()
        local_relation_counter_lc = Counter()
        local_sentences_list = list()
        local_sentences_list_lc = list()

        for j, triple in enumerate(reader):

            if triple['polarity'] != 'POSITIVE':
                continue

            result = process_triple(j, AVRO_FILE, triple, start_time, len(local_entity_mentions_map), self.job)
            if result is None:
                continue

            subject_word, relation_word, object_word, subject_word_lc, relation_word_lc, object_word_lc = result

            if subject_word == object_word:
                continue

            subject_wiki_link = normalize_wiki_entity([w['w_link']['wiki_link'] for w in triple['subject']])
            object_wiki_link = normalize_wiki_entity([w['w_link']['wiki_link'] for w in triple['object']])

            if len(subject_wiki_link) == 1:
                if subject_wiki_link[0] not in local_entity_mentions_map:
                    local_entity_mentions_map[subject_wiki_link[0]] = Counter()
                    local_entity_mentions_map_lc[subject_wiki_link[0]] = Counter()
                local_entity_mentions_map[subject_wiki_link[0]][tuple(subject_word)] += 1
                local_entity_mentions_map_lc[subject_wiki_link[0]][tuple(subject_word_lc)] += 1
            if len(object_wiki_link) == 1:
                if object_wiki_link[0] not in local_entity_mentions_map:
                    local_entity_mentions_map[object_wiki_link[0]] = Counter()
                    local_entity_mentions_map_lc[object_wiki_link[0]] = Counter()
                local_entity_mentions_map[object_wiki_link[0]][tuple(object_word)] += 1
                local_entity_mentions_map_lc[object_wiki_link[0]][tuple(object_word_lc)] += 1

                # if 'NO_WIKI_LINK' not in local_entity_mentions_map:
                #     local_entity_mentions_map[object_wiki_link[0]] = Counter()
                #     local_entity_mentions_map_lc[object_wiki_link[0]] = Counter()
                # local_entity_mentions_map[object_wiki_link[0]][tuple(object_word)] += 1
                # local_entity_mentions_map_lc[object_wiki_link[0]][tuple(object_word_lc)] += 1


            if len(relation_word) > self.job.opts.process_avro__len_relation_word or \
                    len(relation_word) == 0:
                continue

            subject_wiki_link = normalize_wiki_entity([w['w_link']['wiki_link'] for w in triple['subject']])
            object_wiki_link = normalize_wiki_entity([w['w_link']['wiki_link'] for w in triple['object']])

            map_to_tag_dict = dict()
            for w in sorted(triple['relation'] + triple['dropped_words_relation'], key=lambda x: x['index']):
                map_to_tag_dict[w['index']] = '[REL]'
            for w in sorted(triple['subject'] + triple['dropped_words_subject'], key=lambda x: x['index']):
                map_to_tag_dict[w['index']] = '[SUBJ]'
            for w in sorted(triple['object'] + triple['dropped_words_object'], key=lambda x: x['index']):
                map_to_tag_dict[w['index']] = '[OBJ]'

            sentence = [w['word'] for w in sorted(triple['sentence_linked']['tokens'], key=lambda x: x['index'])]
            sentence_lc = [w['word'].lower() for w in sorted(triple['sentence_linked']['tokens'], key=lambda x: x['index'])]

            sentence_mask = ['-' if w['index'] not in map_to_tag_dict else map_to_tag_dict[w['index']] for w in sorted(triple['sentence_linked']['tokens'], key=lambda x: x['index'])]

            local_triples_list.append(
                (
                    (subject_word, relation_word, object_word),
                    (subject_wiki_link[0] if len(subject_wiki_link) > 0 else None, object_wiki_link[0] if len(object_wiki_link) > 0 else None),
                    (triple['triple_id'], triple['article_id'],)
                )
            )

            local_sentences_list.append((triple['triple_id'], (sentence, sentence_mask)))

            local_triples_list_lc.append(
                (
                    (subject_word_lc, relation_word_lc, object_word_lc),
                    (subject_wiki_link[0] if len(subject_wiki_link) > 0 else None, object_wiki_link[0] if len(object_wiki_link) > 0 else None),
                    (triple['triple_id'], triple['article_id'],)
                )
            )

            local_sentences_list_lc.append((triple['triple_id'], (sentence_lc, sentence_mask)))

            local_relation_counter[tuple(relation_word)] += 1
            local_relation_counter_lc[tuple(relation_word_lc)] += 1

        self.job.log('Finished {} triples processed {} entities collected {} sec'.format(AVRO_FILE, len(local_entity_mentions_map), (time.time() - start_time)))
        reader.close()
        return local_entity_mentions_map, local_entity_mentions_map_lc, local_triples_list, local_relation_counter, local_triples_list_lc, local_relation_counter_lc, local_sentences_list, local_sentences_list_lc


class ProcessAVRO(PipelineJob):
    """
    Collect mention entity counts from the Wikiextractor files.
    """
    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                f"data/{opts.process_avro__opiec_dir}/",
            ],
            provides=[
                f"data/versions/{opts.data_version_name}/indexes/entity_mentions_map.dict.pickle",
                f"data/versions/{opts.data_version_name}/indexes/entity_mentions_map_clean.dict.pickle",
                f"data/versions/{opts.data_version_name}/indexes/entity_mentions_map_clean_lc.dict.pickle",
                f"data/versions/{opts.data_version_name}/indexes/triples_list.list.pickle",
                f"data/versions/{opts.data_version_name}/indexes/relation_counter.dict.pickle",
                f"data/versions/{opts.data_version_name}/indexes/triples_list_lc.list.pickle",
                f"data/versions/{opts.data_version_name}/indexes/relation_counter_lc.dict.pickle",
                f"data/versions/{opts.data_version_name}/indexes/sentences_list.pickle",
                f"data/versions/{opts.data_version_name}/indexes/sentences_lc_list.pickle",
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        in_queue = multiprocessing.Queue()
        out_queue = multiprocessing.Queue()

        workers = list()

        #
        # start the workers in individual processes
        #
        for id in range(self.opts.process_avro__nr_of_workers):
            worker = Worker(self, in_queue, out_queue)
            worker.start()
            workers.append(worker)

        # fill the queue
        for file_nr, AVRO_FILE in enumerate(glob.glob(f"data/{self.opts.process_avro__opiec_dir}/part-r-*.avro")):
            in_queue.put(AVRO_FILE)
            if self.opts.process_avro__debug_file_nr == file_nr: break

        outputs = list()

        # collect the output
        for file_nr, AVRO_FILE in enumerate(glob.glob(f"data/{self.opts.process_avro__opiec_dir}/part-r-*.avro")):
            outputs.append(out_queue.get())
            if self.opts.process_avro__debug_file_nr == file_nr: break

        entity_mentions_map = dict()
        entity_mentions_map_lc = dict()
        triples_list = list()
        relation_counter = Counter()
        triples_list_lc = list()
        relation_counter_lc = Counter()
        sentences_list = list()
        sentences_lc_list = list()

        for (
            worker_entity_mentions_map, worker_entity_mentions_map_lc, worker_triples_list, worker_relation_counter, worker_triples_list_lc, worker_relation_counter_lc, worker_sentences_list, worker_sentences_lc_list), AVRO_FILE in outputs:

            for k, v in worker_entity_mentions_map.items():
                if k not in entity_mentions_map:
                    entity_mentions_map[k] = v
                else:
                    entity_mentions_map[k] += v

            for k, v in worker_entity_mentions_map_lc.items():
                if k not in entity_mentions_map_lc:
                    entity_mentions_map_lc[k] = v
                else:
                    entity_mentions_map_lc[k] += v

            triples_list.extend(worker_triples_list)
            triples_list_lc.extend(worker_triples_list_lc)

            sentences_list.extend(worker_sentences_list)
            sentences_lc_list.extend(worker_sentences_lc_list)

            relation_counter += worker_relation_counter
            relation_counter_lc += worker_relation_counter_lc

        # put the None into the queue so the loop in the run() function of the worker stops
        for worker in workers:
            in_queue.put(None)
            out_queue.put(None)

        # terminate the process
        for worker in workers:
            worker.join()

        entity_mentions_map_clean = [
            (
                k,
                [
                    (mention, count) for (mention, count) in ent_mention_counts.most_common(self.opts.process_avro__the_top_k_mentions_per_entity)
                    if count > self.opts.process_avro__min_mention_occurrence_count]
            )

            for k, ent_mention_counts in sorted(
                entity_mentions_map.items(),
                key=lambda x: sum([j for i, j in x[1].items()]),
                reverse=True
            )
            if sum([j for i, j in ent_mention_counts.items()]) >= self.opts.process_avro__minimum_total_count_for_mentions_per_entity

        ]

        entity_mentions_map_clean_lc = [
            (k, [(mention, count) for (mention, count) in v.most_common(self.opts.process_avro__the_top_k_mentions_per_entity) if count > self.opts.process_avro__min_mention_occurrence_count])
            for k, v in sorted(
                entity_mentions_map_lc.items(),
                key=lambda x: sum([j for i, j in x[1].items()]),
                reverse=True
            )
            if sum([j for i, j in v.items()]) >= self.opts.process_avro__minimum_total_count_for_mentions_per_entity
        ]


        with open(f'data/versions/{self.opts.data_version_name}/indexes/entity_mentions_map.dict.pickle', 'wb') as f:
            pickle.dump(entity_mentions_map, f)

        with open(f'data/versions/{self.opts.data_version_name}/indexes/entity_mentions_map_clean.dict.pickle', 'wb') as f:
            pickle.dump(entity_mentions_map_clean, f)

        with open(f'data/versions/{self.opts.data_version_name}/indexes/entity_mentions_map_clean_lc.dict.pickle', 'wb') as f:
            pickle.dump(entity_mentions_map_clean_lc, f)

        with open(f'data/versions/{self.opts.data_version_name}/indexes/triples_list.list.pickle', 'wb') as f:
            pickle.dump(triples_list, f)

        with open(f'data/versions/{self.opts.data_version_name}/indexes/relation_counter.dict.pickle', 'wb') as f:
            pickle.dump(relation_counter, f)

        with open(f'data/versions/{self.opts.data_version_name}/indexes/triples_list_lc.list.pickle', 'wb') as f:
            pickle.dump(triples_list_lc, f)

        with open(f'data/versions/{self.opts.data_version_name}/indexes/relation_counter_lc.dict.pickle', 'wb') as f:
            pickle.dump(relation_counter_lc, f)

        with open(f'data/versions/{self.opts.data_version_name}/indexes/sentences_list.pickle', 'wb') as f:
            pickle.dump(sentences_list, f)

        with open(f'data/versions/{self.opts.data_version_name}/indexes/sentences_lc_list.pickle', 'wb') as f:
            pickle.dump(sentences_lc_list, f)

