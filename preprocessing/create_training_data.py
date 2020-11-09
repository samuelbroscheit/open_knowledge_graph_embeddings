import pickle
import sys
from typing import Dict

import elasticsearch
import time
import tqdm

sys.setrecursionlimit(10000)

from preprocessing.pipeline_job import PipelineJob


def some_query_terms_entity_pair(
    triple, hits, es, INDEX_NAME, filter_stopwords, entity_mentions_map_filtered_low_count_implicits_dict
):
    #     logger.info(q1, '#' ,q2)
    def _internal_func(q1, field):
        result = es.search(
            index=INDEX_NAME,
            size=hits,
            body={"query": {"bool": {"must": [{"match": {field: w}} for w in filter_stopwords(q1.split())]}}},
        )
        return [
            (
                h["_source"]["subject_mention"],
                h["_source"]["relation"],
                h["_source"]["object_mention"],
                h["_source"]["triple_id"],
            )
            for h in sorted(result["hits"]["hits"], key=lambda x: x["_score"], reverse=True)
        ]

    q1 = triple[0][0]
    r = triple[0][1]
    q2 = triple[0][2]
    all_pairs = []
    q1_stack = [q1]
    q2_stack = [q2]
    if triple[1][0] is not None and triple[1][0] in entity_mentions_map_filtered_low_count_implicits_dict:
        q1_stack.extend(entity_mentions_map_filtered_low_count_implicits_dict[triple[1][0]])
    if triple[1][1] is not None and triple[1][1] in entity_mentions_map_filtered_low_count_implicits_dict:
        q2_stack.extend(entity_mentions_map_filtered_low_count_implicits_dict[triple[1][1]])

    for q1_mention in q1_stack:
        for q2_mention in q2_stack:
            all_pairs.append(
                (" ".join(filter_stopwords(q1_mention) + filter_stopwords(q2_mention)), "subject_mention_filt")
            )
            all_pairs.append(
                (" ".join(filter_stopwords(q1_mention) + filter_stopwords(q2_mention)), "object_mention_filt")
            )
    all_pairs = set(all_pairs)
    result = list()
    for q1, field in all_pairs:
        _inter_func_res = _internal_func(q1, field)
        if len(_inter_func_res) > 0:
            result.extend(_inter_func_res)
    return set(result)


def some_query_match_entity_pair(
    triple, hits, es, INDEX_NAME, filter_stopwords, entity_mentions_map_filtered_low_count_implicits_dict
):
    #     logger.info(q1, '#' ,q2)
    def _internal_func(q1, q2):
        result = es.search(
            index=INDEX_NAME,
            size=hits,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"match_phrase": {"subject_mention_filt": q1}},
                            {"match_phrase": {"object_mention_filt": q2}},
                        ]
                    }
                }
            },
        )
        return [
            (
                h["_source"]["subject_mention"],
                h["_source"]["relation"],
                h["_source"]["object_mention"],
                h["_source"]["triple_id"],
            )
            for h in sorted(result["hits"]["hits"], key=lambda x: x["_score"], reverse=True)
        ]

    q1 = triple[0][0]
    r = triple[0][1]
    q2 = triple[0][2]
    all_pairs = []
    q1_stack = [q1]
    q2_stack = [q2]
    if triple[1][0] is not None and triple[1][0] in entity_mentions_map_filtered_low_count_implicits_dict:
        q1_stack.extend(entity_mentions_map_filtered_low_count_implicits_dict[triple[1][0]])
    if triple[1][1] is not None and triple[1][1] in entity_mentions_map_filtered_low_count_implicits_dict:
        q2_stack.extend(entity_mentions_map_filtered_low_count_implicits_dict[triple[1][1]])

    for q1_mention in q1_stack:
        for q2_mention in q2_stack:
            all_pairs.append((" ".join(filter_stopwords(q1_mention)), " ".join(filter_stopwords(q2_mention))))
            all_pairs.append((" ".join(filter_stopwords(q2_mention)), " ".join(filter_stopwords(q1_mention))))
    all_pairs = set(all_pairs)
    result = list()
    for q1, q2 in all_pairs:
        _inter_func_res = _internal_func(q1, q2)
        if len(_inter_func_res) > 0:
            result.extend(_inter_func_res)
    return set(result)


def some_query_match_entity_pair_in_relation(
    triple, hits, es, INDEX_NAME, filter_stopwords, entity_mentions_map_filtered_low_count_implicits_dict
):
    #     logger.info(q1, '#' ,q2)
    def _internal_func(q1, q2, filt_field):
        result = es.search(
            index=INDEX_NAME,
            size=hits,
            body={
                "query": {
                    "bool": {"must": [{"match_phrase": {filt_field: q1}}, {"match_phrase": {"relation_filt": q2}}]}
                }
            },
        )
        return [
            (
                h["_source"]["subject_mention"],
                h["_source"]["relation"],
                h["_source"]["object_mention"],
                h["_source"]["triple_id"],
            )
            for h in sorted(result["hits"]["hits"], key=lambda x: x["_score"], reverse=True)
        ]

    q1 = triple[0][0]
    r = triple[0][1]
    q2 = triple[0][2]
    all_pairs = []
    q1_stack = [q1]
    q2_stack = [q2]
    if triple[1][0] is not None and triple[1][0] in entity_mentions_map_filtered_low_count_implicits_dict:
        q1_stack.extend(entity_mentions_map_filtered_low_count_implicits_dict[triple[1][0]])
    if triple[1][1] is not None and triple[1][1] in entity_mentions_map_filtered_low_count_implicits_dict:
        q2_stack.extend(entity_mentions_map_filtered_low_count_implicits_dict[triple[1][1]])

    for q1_mention in q1_stack:
        for q2_mention in q2_stack:
            all_pairs.append((" ".join(filter_stopwords(q1_mention)), " ".join(filter_stopwords(q2_mention))))
            all_pairs.append((" ".join(filter_stopwords(q2_mention)), " ".join(filter_stopwords(q1_mention))))
    all_pairs = set(all_pairs)
    result = list()
    for q1, q2 in all_pairs:
        _inter_func_res = _internal_func(q1, q2, "subject_mention_filt")
        if len(_inter_func_res) > 0:
            result.extend(_inter_func_res)
        _inter_func_res = _internal_func(q1, q2, "object_mention_filt")
        if len(_inter_func_res) > 0:
            result.extend(_inter_func_res)
    return set(result)


def some_query_match_entity_relation_pair(
    triple, hits, es, INDEX_NAME, filter_stopwords, entity_mentions_map_filtered_low_count_implicits_dict
):
    #     logger.info(q1, '#' ,q2)
    def _internal_func(q1, r, filt_field="subject_mention_filt"):
        result = es.search(
            index=INDEX_NAME,
            size=hits,
            body={
                "query": {
                    "bool": {"must": [{"match_phrase": {filt_field: q1}}, {"match_phrase": {"relation_filt": r}}]}
                }
            },
        )
        return [
            (
                h["_source"]["subject_mention"],
                h["_source"]["relation"],
                h["_source"]["object_mention"],
                h["_source"]["triple_id"],
            )
            for h in sorted(result["hits"]["hits"], key=lambda x: x["_score"], reverse=True)
        ]

    q1, r, q2 = triple[0]
    all_pairs = []
    q1_stack = [q1, q2]
    r_stack = [r]
    e1, e2 = triple[1]
    if e1 is not None and e1 in entity_mentions_map_filtered_low_count_implicits_dict:
        q1_stack.extend(entity_mentions_map_filtered_low_count_implicits_dict[e1])
    if e2 is not None and e2 in entity_mentions_map_filtered_low_count_implicits_dict:
        q1_stack.extend(entity_mentions_map_filtered_low_count_implicits_dict[e2])

    for q1_mention in q1_stack:
        for r_mention in r_stack:
            all_pairs.append(
                (" ".join(filter_stopwords(q1_mention)), " ".join(filter_stopwords(r_mention)), "subject_mention_filt")
            )
            all_pairs.append(
                (" ".join(filter_stopwords(q1_mention)), " ".join(filter_stopwords(r_mention)), "object_mention_filt")
            )
    all_pairs = set(all_pairs)
    result = list()
    for q1, r, field in all_pairs:
        _inter_func_res = _internal_func(q1, r, field)
        if len(_inter_func_res) > 0:
            result.extend(_inter_func_res)
    return set(result)


def some_query_match_entity(
    triple, hits, es, INDEX_NAME, filter_stopwords, entity_mentions_map_filtered_low_count_implicits_dict
):
    #     logger.info(q1, '#' ,q2)
    def _internal_func(q1, filt_field):
        print(q1, filt_field)
        result = es.search(
            index=INDEX_NAME, size=hits, body={"query": {"bool": {"must": [{"match_phrase": {filt_field: q1}},]}}}
        )
        return [
            (
                h["_source"]["subject_mention"],
                h["_source"]["relation"],
                h["_source"]["object_mention"],
                h["_source"]["triple_id"],
            )
            for h in sorted(result["hits"]["hits"], key=lambda x: x["_score"], reverse=True)
        ]

    q1, r, q2 = triple[0]
    all_pairs = []
    q1_stack = [q1, q2]
    e1, e2 = triple[1]
    if e1 is not None and e1 in entity_mentions_map_filtered_low_count_implicits_dict:
        q1_stack.extend(entity_mentions_map_filtered_low_count_implicits_dict[e1])
    if e2 is not None and e2 in entity_mentions_map_filtered_low_count_implicits_dict:
        q1_stack.extend(entity_mentions_map_filtered_low_count_implicits_dict[e2])

    for q1_mention in q1_stack:
        all_pairs.append((" ".join(filter_stopwords(q1_mention)), "subject_mention_filt"))
        all_pairs.append((" ".join(filter_stopwords(q1_mention)), "object_mention_filt"))
    all_pairs = set(all_pairs)
    result = list()
    print(all_pairs)
    for q1, field in all_pairs:
        _inter_func_res = _internal_func(q1, field)
        if len(_inter_func_res) > 0:
            result.extend(_inter_func_res)
    return set(result)


def some_query_exact_entity_pair(
    triple, hits, es, INDEX_NAME, filter_stopwords, entity_mentions_map_filtered_low_count_implicits_dict
):
    #     logger.info(q1, '#' ,q2)
    def _internal_func(q1, q2):
        result = es.search(
            index=INDEX_NAME,
            size=hits,
            body={
                "query": {
                    "bool": {"must": [{"term": {"subject_mention_exact": q1}}, {"term": {"object_mention_exact": q2}}]}
                }
            },
        )
        return [
            (
                h["_source"]["subject_mention"],
                h["_source"]["relation"],
                h["_source"]["object_mention"],
                h["_source"]["triple_id"],
            )
            for h in sorted(result["hits"]["hits"], key=lambda x: x["_score"], reverse=True)
        ]

    q1 = triple[0][0]
    r = triple[0][1]
    q2 = triple[0][2]
    all_pairs = []
    q1_stack = [q1]
    q2_stack = [q2]
    if triple[1][0] is not None and triple[1][0] in entity_mentions_map_filtered_low_count_implicits_dict:
        q1_stack.extend(entity_mentions_map_filtered_low_count_implicits_dict[triple[1][0]])
    if triple[1][1] is not None and triple[1][1] in entity_mentions_map_filtered_low_count_implicits_dict:
        q2_stack.extend(entity_mentions_map_filtered_low_count_implicits_dict[triple[1][1]])

    for q1_mention in q1_stack:
        for q2_mention in q2_stack:
            all_pairs.append((" ".join(filter_stopwords(q1_mention)), " ".join(filter_stopwords(q2_mention))))
            all_pairs.append((" ".join(filter_stopwords(q2_mention)), " ".join(filter_stopwords(q1_mention))))
    all_pairs = set(all_pairs)
    result = list()
    for q1, q2 in all_pairs:
        _inter_func_res = _internal_func(q1, q2)
        if len(_inter_func_res) > 0:
            result.extend(_inter_func_res)
    return set(result)


def some_query_full_triple(
    triple, hits, es, INDEX_NAME, filter_stopwords, entity_mentions_map_filtered_low_count_implicits_dict
):
    #     logger.info(q1, '#' ,q2)
    r = " ".join(triple[0][1])

    def _internal_func(q1, q2):
        result = es.search(
            index=INDEX_NAME,
            size=hits,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"subject_mention_exact": q1}},
                            {"match": {"relation": r}},
                            {"term": {"object_mention_exact": q2}},
                        ]
                    }
                }
            },
        )
        return [
            (
                h["_source"]["subject_mention"],
                h["_source"]["relation"],
                h["_source"]["object_mention"],
                h["_source"]["triple_id"],
            )
            for h in sorted(result["hits"]["hits"], key=lambda x: x["_score"], reverse=True)
        ]

    q1 = triple[0][0]
    q2 = triple[0][2]
    all_pairs = []
    q1_stack = [q1]
    q2_stack = [q2]
    if triple[1][0] is not None and triple[1][0] in entity_mentions_map_filtered_low_count_implicits_dict:
        q1_stack.extend(entity_mentions_map_filtered_low_count_implicits_dict[triple[1][0]])
    if triple[1][1] is not None and triple[1][1] in entity_mentions_map_filtered_low_count_implicits_dict:
        q2_stack.extend(entity_mentions_map_filtered_low_count_implicits_dict[triple[1][1]])

    for q1_mention in q1_stack:
        for q2_mention in q2_stack:
            all_pairs.append((" ".join(filter_stopwords(q1_mention)), " ".join(filter_stopwords(q2_mention))))
            all_pairs.append((" ".join(filter_stopwords(q2_mention)), " ".join(filter_stopwords(q1_mention))))
    all_pairs = set(all_pairs)
    result = list()
    for q1, q2 in all_pairs:
        _inter_func_res = _internal_func(q1, q2)
        if len(_inter_func_res) > 0:
            result.extend(_inter_func_res)
    return set(result)


class CreateTrainingData(PipelineJob):
    """

    """

    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                f"data/versions/{opts.data_version_name}/elasticsearch_index_created",
                f"data/versions/{opts.data_version_name}/indexes/triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations.pickle",
                f"data/versions/{opts.data_version_name}/indexes/entity_mentions_map_filtered_low_count_implicits_dict.pickle",
                f"data/versions/{opts.data_version_name}/indexes/mention_token_dict_most_common_dict.pickle",
                f"data/versions/{opts.data_version_name}/indexes/relation_token_dict_most_common_dict.pickle",
                f"data/versions/{opts.data_version_name}/indexes/validation_data.pickle",
                f"data/versions/{opts.data_version_name}/indexes/validation_linked_data.pickle",
                f"data/versions/{opts.data_version_name}/indexes/test_data.pickle",
                f"data/versions/{opts.data_version_name}/indexes/validation_data_ids.pickle",
                f"data/versions/{opts.data_version_name}/indexes/validation_linked_data_ids.pickle",
                f"data/versions/{opts.data_version_name}/indexes/test_data_ids.pickle",
            ],
            provides=[
                f"data/versions/{opts.data_version_name}/train_data_simple.txt",
                f"data/versions/{opts.data_version_name}/train_data_basic.txt",
                f"data/versions/{opts.data_version_name}/train_data_thorough.txt",
                f"data/versions/{opts.data_version_name}/validation_data.txt",
                f"data/versions/{opts.data_version_name}/validation_data_linked.txt",
                f"data/versions/{opts.data_version_name}/validation_data_linked_no_mention.txt",
                f"data/versions/{opts.data_version_name}/test_data.txt",
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations.pickle",
            "rb",
        ) as f:
            triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations = pickle.load(f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/entity_mentions_map_filtered_low_count_implicits_dict.pickle",
            "rb",
        ) as f:
            entity_mentions_map_filtered_low_count_implicits_dict = pickle.load(f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/mention_token_dict_most_common_dict.pickle", "rb"
        ) as f:
            mention_token_dict_most_common_dict = pickle.load(f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/relation_token_dict_most_common_dict.pickle", "rb"
        ) as f:
            relation_token_dict_most_common_dict = pickle.load(f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/validation_data.pickle", "rb",) as f:
            validation_data = pickle.load(f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/validation_linked_data.pickle", "rb",) as f:
            validation_linked_data = pickle.load(f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/test_data.pickle", "rb",) as f:
            test_data = pickle.load(f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/validation_data_ids.pickle", "rb",) as f:
            validation_data_ids = pickle.load(f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/validation_linked_data_ids.pickle", "rb",) as f:
            validation_linked_data_ids = pickle.load(f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/test_data_ids.pickle", "rb",) as f:
            test_data_ids = pickle.load(f)

        stopwords = list(mention_token_dict_most_common_dict.keys())[:25]
        stopwords += list(relation_token_dict_most_common_dict.keys())[:25]
        stopwords = set(stopwords)

        def filter_stopwords(toks):
            result = tuple([t for t in toks if t not in stopwords])
            if len(result) > 0:
                return result
            return toks

        INDEX_NAME = "training_triples_mentions"

        es = elasticsearch.Elasticsearch([{"host": self.opts.create_elasticsearch_index__host, "port": 9200}])

        filter_ids_for_train_data_simple = set()
        filter_ids_for_train_data_thorough = set()

        for data in [test_data, validation_data, validation_linked_data]:
            for j, ((s, r, o), (se, oe)) in enumerate(tqdm.tqdm(data)):

                # easy
                for _, _, _, triple_id in some_query_full_triple(
                    ((s, r, o), (se, oe)),
                    1000,
                    es,
                    INDEX_NAME,
                    filter_stopwords,
                    entity_mentions_map_filtered_low_count_implicits_dict,
                ):
                    filter_ids_for_train_data_simple.add(triple_id)

                res = some_query_match_entity_pair(
                    ((s, r, o), (se, oe)),
                    1000,
                    es,
                    INDEX_NAME,
                    filter_stopwords,
                    entity_mentions_map_filtered_low_count_implicits_dict,
                )
                for _, _, _, triple_id in res:
                    filter_ids_for_train_data_thorough.add(triple_id)

                res = some_query_terms_entity_pair(
                    ((s, r, o), (se, oe)),
                    1000,
                    es,
                    INDEX_NAME,
                    filter_stopwords,
                    entity_mentions_map_filtered_low_count_implicits_dict,
                )
                if len(res) < 1000:
                    for _, _, _, triple_id in res:
                        filter_ids_for_train_data_thorough.add(triple_id)

                res = some_query_match_entity_pair_in_relation(
                    ((s, r, o), (se, oe)),
                    1000,
                    es,
                    INDEX_NAME,
                    filter_stopwords,
                    entity_mentions_map_filtered_low_count_implicits_dict,
                )
                if len(res) < 1000:
                    for _, _, _, triple_id in res:
                        filter_ids_for_train_data_thorough.add(triple_id)

        evaluation_ids = test_data_ids
        evaluation_ids.extend(validation_data_ids)
        evaluation_ids.extend(validation_linked_data_ids)
        evaluation_ids = set(evaluation_ids)

        # # ############# Create training data  ###############
        # # logger.info("Create training data")

        train_data_simple = list()
        train_data_basic = list()
        train_data_thorough = list()

        # start = time.time()

        for i, ((s, r, o), (se, oe)) in enumerate(
            tqdm.tqdm(triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations)
        ):

            if i not in evaluation_ids:
                train_data_simple.append(((s, r, o), (se, oe)))

            if i not in filter_ids_for_train_data_simple:
                train_data_basic.append(((s, r, o), (se, oe)))

            if i not in filter_ids_for_train_data_thorough and i not in filter_ids_for_train_data_simple:
                train_data_thorough.append(((s, r, o), (se, oe)))

        with open(f"data/versions/{self.opts.data_version_name}/train_data_simple.txt", "w") as f:
            for ((s, r, o), (se, oe)) in train_data_simple:
                f.writelines(
                    "{}\t{}\t{}\t{}\t{}\n".format(" ".join(s), " ".join(r), " ".join(o), " ".join(s), " ".join(o),)
                )

        with open(f"data/versions/{self.opts.data_version_name}/train_data_basic.txt", "w") as f:
            for ((s, r, o), (se, oe)) in train_data_basic:
                f.writelines(
                    "{}\t{}\t{}\t{}\t{}\n".format(" ".join(s), " ".join(r), " ".join(o), " ".join(s), " ".join(o),)
                )

        with open(f"data/versions/{self.opts.data_version_name}/train_data_thorough.txt", "w") as f:
            for ((s, r, o), (se, oe)) in train_data_thorough:
                f.writelines(
                    "{}\t{}\t{}\t{}\t{}\n".format(" ".join(s), " ".join(r), " ".join(o), " ".join(s), " ".join(o),)
                )

        def get_mentions_for_entity(e, m_default):
            return (
                list(
                    set(
                        [" ".join(m) for m, c in entity_mentions_map_filtered_low_count_implicits_dict[e].items()]
                        + [" ".join(m_default)]
                    )
                )
                if e in entity_mentions_map_filtered_low_count_implicits_dict
                and len(entity_mentions_map_filtered_low_count_implicits_dict[e]) > 0
                else [" ".join(m_default)]
            )

        with open(f"data/versions/{self.opts.data_version_name}/validation_data.txt", "w") as f:
            for ((s, r, o), (se, oe)) in validation_data:
                f.writelines(
                    "{}\t{}\t{}\t{}\t{}\n".format(" ".join(s), " ".join(r), " ".join(o), " ".join(s), " ".join(o),)
                )

        with open(f"data/versions/{self.opts.data_version_name}/validation_data_linked.txt", "w") as f:
            for ((s, r, o), (se, oe)) in validation_linked_data:
                f.writelines(
                    "{}\t{}\t{}\t{}\t{}\n".format(
                        " ".join(s),
                        " ".join(r),
                        " ".join(o),
                        "|||".join(get_mentions_for_entity(se, s)),
                        "|||".join(get_mentions_for_entity(oe, o)),
                    )
                )

        with open(f"data/versions/{self.opts.data_version_name}/validation_data_linked_no_mention.txt", "w") as f:
            for ((s, r, o), (se, oe)) in validation_linked_data:
                f.writelines(
                    "{}\t{}\t{}\t{}\t{}\n".format(" ".join(s), " ".join(r), " ".join(o), " ".join(s), " ".join(o),)
                )

        with open(f"data/versions/{self.opts.data_version_name}/test_data.txt", "w") as f:
            for ((s, r, o), (se, oe)) in test_data:
                f.writelines(
                    "{}\t{}\t{}\t{}\t{}\n".format(
                        " ".join(s),
                        " ".join(r),
                        " ".join(o),
                        "|||".join(get_mentions_for_entity(se, s)),
                        "|||".join(get_mentions_for_entity(oe, o)),
                    )
                )

        # with open(f"data/versions/{self.opts.data_version_name}/elasticsearch_index_created", "w",) as f:
        #     f.writelines(["SUCCESS"])
