import math
import numpy
import pickle
import sys
from collections import Counter, OrderedDict
from typing import Dict

import tqdm

from preprocessing.misc import merge

sys.setrecursionlimit(10000)

from preprocessing.pipeline_job import PipelineJob


class SampleEvaluation(PipelineJob):
    """

    """

    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                f'data/versions/{opts.data_version_name}/indexes/triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations.pickle',
            ],
            provides=[
                f"data/versions/{opts.data_version_name}/indexes/validation_data.pickle",
                f"data/versions/{opts.data_version_name}/indexes/validation_linked_data.pickle",
                f"data/versions/{opts.data_version_name}/indexes/test_data.pickle",
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        with open(f'data/versions/{self.opts.data_version_name}/indexes/triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations.pickle', 'rb') as f:
            triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations = pickle.load(f)


        triple_all_with_specific_relations = [
            (i, ((s, r, o), (se, oe))) for i, ((s, r, o), (se, oe)) in enumerate(triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations)
            if len(r) >= self.opts.sample_evaluation__min_relation_token_length_for_testing
        ]

        triple_all_with_specific_relations_ids = list()
        triple_all_with_specific_relations_linked_ids = list()
        triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations_filtered_for_test_dict = dict()

        for i, ((s, r, o), (se, oe)) in tqdm.tqdm(triple_all_with_specific_relations):
            triple_all_with_specific_relations_ids.append(i)
            if se is not None and oe is not None:
                triple_all_with_specific_relations_linked_ids.append(i)
                triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations_filtered_for_test_dict[i] = (((s, r, o), (se, oe)))

        # sample evaluation data

        linked_eval_data_ids = sorted(numpy.random.choice(triple_all_with_specific_relations_linked_ids, (2*self.opts.sample_evaluation__eval_data_size+20), replace =False))
        linked_eval_data_ids = set(linked_eval_data_ids)

        while True:

            validation_data_ids = sorted(numpy.random.choice(triple_all_with_specific_relations_ids, (self.opts.sample_evaluation__eval_data_size+100), replace=False))
            validation_data_ids = set(validation_data_ids)

            validation_data_ids.difference_update(linked_eval_data_ids)

            overlap = len(linked_eval_data_ids.intersection(validation_data_ids))

            if (overlap == 0 and
                len(linked_eval_data_ids) >= self.opts.sample_evaluation__eval_data_size and
                len(validation_data_ids) >= self.opts.sample_evaluation__eval_data_size
               ):
                break

        linked_eval_data_ids = list(linked_eval_data_ids)[:2*self.opts.sample_evaluation__eval_data_size]
        test_data_ids, validation_linked_data_ids = linked_eval_data_ids[:self.opts.sample_evaluation__eval_data_size], linked_eval_data_ids[self.opts.sample_evaluation__eval_data_size:]
        validation_data_ids = list(validation_data_ids)[:self.opts.sample_evaluation__eval_data_size]

        self.log("Finished")

        validation_data = list(map(triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations.__getitem__, validation_data_ids))
        test_data = list(map(triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations_filtered_for_test_dict.__getitem__, test_data_ids))
        validation_linked_data = list(map(triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations_filtered_for_test_dict.__getitem__, validation_linked_data_ids))

        with open(f"data/versions/{self.opts.data_version_name}/indexes/validation_data.pickle", "wb",) as f:
            pickle.dump(validation_data, f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/validation_linked_data.pickle", "wb",) as f:
            pickle.dump(validation_linked_data, f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/test_data.pickle", "wb",) as f:
            pickle.dump(test_data, f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/validation_data_ids.pickle", "wb",) as f:
            pickle.dump(validation_data_ids, f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/validation_linked_data_ids.pickle", "wb",) as f:
            pickle.dump(validation_linked_data_ids, f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/test_data_ids.pickle", "wb",) as f:
            pickle.dump(test_data_ids, f)

