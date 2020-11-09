import pickle
import sys
from typing import Dict

import elasticsearch
import time
import tqdm

sys.setrecursionlimit(10000)

from preprocessing.pipeline_job import PipelineJob


class CreateElasticsSearch(PipelineJob):
    """

    """

    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                f'data/versions/{opts.data_version_name}/indexes/triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations.pickle',
                f'data/versions/{opts.data_version_name}/indexes/mention_token_dict_most_common_dict.pickle',
                f'data/versions/{opts.data_version_name}/indexes/relation_token_dict_most_common_dict.pickle',
            ],
            provides=[
                f"data/versions/{opts.data_version_name}/elasticsearch_index_created",
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        with open(f'data/versions/{self.opts.data_version_name}/indexes/triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations.pickle', 'rb') as f:
            triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations = pickle.load(f)

        with open(f'data/versions/{self.opts.data_version_name}/indexes/mention_token_dict_most_common_dict.pickle', 'rb') as f:
            mention_token_dict_most_common_dict = pickle.load(f)

        with open(f'data/versions/{self.opts.data_version_name}/indexes/relation_token_dict_most_common_dict.pickle', 'rb') as f:
            relation_token_dict_most_common_dict = pickle.load(f)

        stopwords = list(mention_token_dict_most_common_dict.keys())[:25]
        stopwords += list(relation_token_dict_most_common_dict.keys())[:25]
        stopwords = set(stopwords)

        def filter_stopwords(toks):
            result = tuple([t for t in toks if t not in stopwords])
            if len(result) > 0: return result
            return toks

        ############# Filter test and validation from training data #########
        self.log("Filter test and validation from training data")


        INDEX_NAME = 'training_triples_mentions'
        TYPE_NAME = 'training_triples_mentions'

        es = elasticsearch.Elasticsearch([{'host': self.opts.create_elasticsearch_index__host, 'port': 9200}])

        if es.indices.exists(INDEX_NAME):
            self.log("{} deleting {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), INDEX_NAME))
            res = es.indices.delete(index=INDEX_NAME, request_timeout=900)

        body = {
          "mappings": {
            INDEX_NAME: {
              "properties": {
                "triple_id": {
                    "type": "integer"
                },
                "subject_mention": {
                    "type": "text"
                },
                "relation": {
                    "type": "text"
                },
                "object_mention": {
                    "type": "text"
                },
                "subject_mention_filt": {
                    "type": "text"
                },
                "relation_filt": {
                    "type": "text"
                },
                "object_mention_filt": {
                    "type": "text"
                },
                "subject_mention_exact": {
                    "type": "keyword"
                },
                "relation_exact": {
                    "type": "keyword"
                },
                "object_mention_exact": {
                    "type": "keyword"
                },
              }
            }
          }
        }

        res = es.indices.create(index = INDEX_NAME, body=body)

        def get_dicts(triple_id ,subject_mention, relation, object_mention):
            op_dict = {
                "index": {
                    "_index": INDEX_NAME,
                    "_type" : TYPE_NAME,
                }
            }
            data_dict = {
                'triple_id': triple_id,
                'subject_mention': ' '.join(subject_mention),
                'relation': ' '.join(relation),
                'object_mention': ' '.join(object_mention),
                'subject_mention_filt': ' '.join(filter_stopwords(subject_mention)),
                'relation_filt': ' '.join(filter_stopwords(relation)),
                'object_mention_filt': ' '.join(filter_stopwords(object_mention)),
                'subject_mention_exact': ' '.join(filter_stopwords(subject_mention)),
                'relation_exact': ' '.join(filter_stopwords(relation)),
                'object_mention_exact': ' '.join(filter_stopwords(object_mention)),
            }
            yield op_dict
            yield data_dict

        self.log("{} Start indexing".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), time.time()))

        bulk = []
        bulksize = 100000
        start = time.time()
        for i, ((s,r,o), (se,oe)) in enumerate(tqdm.tqdm(triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations)):
            for adict in get_dicts(i,s,r,o):
                bulk.append(adict)
            if len(bulk) == bulksize:
                es.bulk(index=INDEX_NAME, body=bulk, refresh=True, request_timeout=900, )
                bulk = []

        self.log("{} Finished indexing".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        with open(f"data/versions/{self.opts.data_version_name}/elasticsearch_index_created", "w",) as f:
            f.writelines(["SUCCESS"])

