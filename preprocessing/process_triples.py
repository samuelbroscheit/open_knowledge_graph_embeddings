import math
import pickle
import sys
from collections import Counter, OrderedDict
from typing import Dict

import tqdm

sys.setrecursionlimit(10000)

from preprocessing.pipeline_job import PipelineJob


class ProcessTriples(PipelineJob):
    """

    """

    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                "data/indexes/redirects_en.ttl.bz2.dict",
                f"data/versions/{opts.data_version_name}/indexes/triples_list_lc.list.pickle",
                f"data/versions/{opts.data_version_name}/indexes/mention_entity_dict.pickle",
            ],
            provides=[
                f'data/versions/{opts.data_version_name}/indexes/triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations.pickle',
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        with open(f'data/versions/{self.opts.data_version_name}/indexes/triples_list_lc.list.pickle', 'rb') as f:
            triples_list_lc = pickle.load(f)

        with open("data/indexes/redirects_en.ttl.bz2.dict", "rb") as f:
            redirects_en = pickle.load(f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/mention_entity_dict.pickle", "rb") as f:
            mention_entity_dict = pickle.load(f)

        def redirect_entity(ent):
            if ent is not None:
                ent_underscore = ent.replace(' ', '_')
                if ent_underscore in redirects_en:
                    ent = redirects_en[ent_underscore].replace('_', ' ')
            return ent

        # Aggregate triples and count subject and object *entities* per mention
        self.log("Aggregate triples and count subject and object entities per mention")

        triples_sro_to_se_oe_dict = OrderedDict()
        len_triples_list_lc = len(triples_list_lc)

        for ((s, r, o), (se, oe), (_,_)) in tqdm.tqdm(triples_list_lc):
            s, r, o = (tuple(s), tuple(r), tuple(o))
            se = redirect_entity(se)
            oe = redirect_entity(oe)
            if (s, r, o) not in triples_sro_to_se_oe_dict:
                triples_sro_to_se_oe_dict[(s, r, o)] = (Counter(), Counter())
            if se is not None:
                triples_sro_to_se_oe_dict[(s, r, o)][0][se] += 1
            if oe is not None:
                triples_sro_to_se_oe_dict[(s, r, o)][1][oe] += 1

        self.log("triples_list_lc {:,}".format(len(triples_list_lc)))
        self.log("triples_sro_to_se_oe_dict {:,}".format(len(triples_sro_to_se_oe_dict)))


        # Make aggregated triples mention entities pure by filtering entitiy annotations
        # for subjects or objects are highly unlikely and therefore most likely wrong

        self.log("Make aggregated mention entities pure")

        def get_confidence_treshold(c):
            tot = sum(c.values())
            return 1 - 1 / math.log(tot) if tot > 1 else 0

        triples_list_lc_unique_most_popular_links = list()
        filtered_triples_list_lc_unique_most_popular_links = list()
        len_triples_sro_to_se_oe_dict = len(triples_sro_to_se_oe_dict)

        for ((s, r, o), (se_counter, oe_counter)) in tqdm.tqdm(triples_sro_to_se_oe_dict.items()):

            se_ok, oe_ok = True, True
            se, oe = None, None
            if len(se_counter) > 0:
                se, se_counts = se_counter.most_common()[0]
                se_pop = se_counts / sum(se_counter.values())
                se_ok = se_pop >= get_confidence_treshold(se_counter)
                if not se_ok:
                    se = None
            if len(oe_counter) > 0:
                oe, oe_counts = oe_counter.most_common()[0]
                oe_pop = oe_counts / sum(oe_counter.values())
                oe_ok = oe_pop >= get_confidence_treshold(oe_counter)
                if not oe_ok:
                    oe = None

            if se is not None and se_ok and oe is not None and oe_ok and se == oe:
                # something is wrong with this triple
                filtered_triples_list_lc_unique_most_popular_links.append(((s, r, o), (se_counter, oe_counter)))
                se, oe = None, None

            triples_list_lc_unique_most_popular_links.append(((s, r, o), (se, oe)))

        self.log("triples_list_lc_unique_most_popular_links {:,}".format(len(filtered_triples_list_lc_unique_most_popular_links)))



        # Filter mentions and relations from aggregated triples by most common tokens
        self.log("Filter aggregated triples by most common tokens")

        mention_token_dict = Counter()
        relation_token_dict = Counter()

        for ((s, r, o), (se_counter, oe_counter)) in tqdm.tqdm(triples_sro_to_se_oe_dict.items()):
            for m in (s, o):
                mention_token_dict.update(m)
            relation_token_dict.update(r)

        mention_token_dict_most_common_dict = OrderedDict(mention_token_dict.most_common(self.opts.process_triples__mention_tokens_vocab_size))
        relation_token_dict_most_common_dict = OrderedDict(relation_token_dict.most_common(self.opts.process_triples__relation_tokens_vocab_size))

        self.log("list(mention_token_dict_most_common_dict.values())[-1]) {:,}".format(list(mention_token_dict_most_common_dict.values())[-1]))
        self.log("list(relation_token_dict_most_common_dict.values())[-1] {:,}".format(list(relation_token_dict_most_common_dict.values())[-1]))

        triples_list_lc_unique_most_popular_links_most_common_toks_list = list()
        mention_entity_dict_most_common_toks = OrderedDict()
        relation_counter_most_common_toks = Counter()
        mention_counter_most_common_toks = Counter()


        # Now remove triples that are underrepresented because of missing tokens and
        # create updates stats for mentions and relations
        self.log("Remove triples that are underrepresented because of missing tokens")
        for ((s, r, o), (se, oe)) in tqdm.tqdm(triples_list_lc_unique_most_popular_links):
            do_continue = False
            for m, _dict in ((s, mention_token_dict_most_common_dict), (r, relation_token_dict_most_common_dict), (o, mention_token_dict_most_common_dict),):
                found = sum(list(map(lambda t: 1 if t in _dict else 0, m)))
                if found < len(m):
                    do_continue = True
                    break
            if do_continue: continue

            triples_list_lc_unique_most_popular_links_most_common_toks_list.append(((s, r, o), (se, oe)))
            if s not in mention_entity_dict_most_common_toks and s in mention_entity_dict:
                mention_entity_dict_most_common_toks[s] = mention_entity_dict[s]
            if o not in mention_entity_dict_most_common_toks and o in mention_entity_dict:
                mention_entity_dict_most_common_toks[o] = mention_entity_dict[o]
            mention_counter_most_common_toks.update((s, o))
            relation_counter_most_common_toks[r] += 1

        mention_counter_most_common_toks_most_common = list((k, v) for k, v in mention_counter_most_common_toks.items() if v > 2)
        mention_counter_most_common_toks_most_common_dict = dict(mention_counter_most_common_toks_most_common)
        relation_counter_most_common_toks_most_common = list((k, v) for k, v in relation_counter_most_common_toks.items() if v > 2)
        relation_counter_most_common_toks_most_common_dict = dict(relation_counter_most_common_toks_most_common)

        self.log("len(mention_counter_most_common_toks_most_common) {:,}".format(len(mention_counter_most_common_toks_most_common)))
        self.log("len(relation_counter_most_common_toks_most_common) {:,}".format(len(relation_counter_most_common_toks_most_common)))


        # Now only keep triples with the remaining relations and mentions
        self.log("Now only keep triples with the remaining relations and mentions")

        del relation_counter_most_common_toks_most_common_dict[('is:impl_appos-clause',)]
        del relation_counter_most_common_toks_most_common_dict[('is:impl_appos-clause', 'in:impl_appos-clause',)]

        self.log("triples_list_lc_unique_most_popular_links {:,}".format(len(triples_list_lc_unique_most_popular_links)))

        triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations = list()
        relation_counter_final_triples = Counter()
        mention_counter_final_triples = Counter()
        len_triples_list_lc = len(triples_list_lc_unique_most_popular_links)

        for (s, r, o), (se, oe) in tqdm.tqdm(triples_list_lc_unique_most_popular_links_most_common_toks_list):
            if (r in relation_counter_most_common_toks_most_common_dict and
                    s in mention_counter_most_common_toks_most_common_dict and
                    o in mention_counter_most_common_toks_most_common_dict
            ):
                triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations.append(((tuple(s), tuple(r), tuple(o)), (se, oe)))
                # if se is not None and oe is not None:
                #     triples_list_lc_unique_most_common_mentions_and_relations_linked.append(((tuple(s), tuple(r), tuple(o)), (se, oe)))
                mention_counter_final_triples[s] += 1
                mention_counter_final_triples[o] += 1
                relation_counter_final_triples[r] += 1

        self.log("triples_list_lc_unique_most_common_mentions_and_relations {:,}".format(len(triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations)))

        with open(f"data/versions/{self.opts.data_version_name}/indexes/triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations.pickle", "wb",) as f:
            pickle.dump(triples_list_lc_unique_most_popular_links_most_common_mentions_and_relations, f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/mention_token_dict_most_common_dict.pickle", "wb",) as f:
            pickle.dump(mention_token_dict_most_common_dict, f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/relation_token_dict_most_common_dict.pickle", "wb",) as f:
            pickle.dump(relation_token_dict_most_common_dict, f)

