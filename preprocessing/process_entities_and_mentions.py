import pickle
import sys
from collections import Counter, OrderedDict
from typing import Dict

import tqdm

from preprocessing.misc import merge

sys.setrecursionlimit(10000)

from preprocessing.pipeline_job import PipelineJob


class ProcessMentionAndEntities(PipelineJob):
    """

    """

    def __init__(self, preprocess_jobs: Dict[str, PipelineJob], opts):
        super().__init__(
            requires=[
                "data/indexes/redirects_en.ttl.bz2.dict",
                f"data/versions/{opts.data_version_name}/indexes/entity_mentions_map_clean_lc.dict.pickle",
            ],
            provides=[
                f"data/versions/{opts.data_version_name}/indexes/mention_entity_dict.pickle",
                f"data/versions/{opts.data_version_name}/indexes/entity_mentions_map_filtered_low_count_implicits_dict.pickle",
            ],
            preprocess_jobs=preprocess_jobs,
            opts=opts,
        )

    def _run(self):

        with open("data/indexes/redirects_en.ttl.bz2.dict", "rb") as f:
            redirects_en = pickle.load(f)

        with open(
            f"data/versions/{self.opts.data_version_name}/indexes/entity_mentions_map_clean_lc.dict.pickle", "rb"
        ) as f:
            entity_mentions_map = pickle.load(f)

        ############ Begin fix entity mention map ############

        redirects_en_lc_spaces = {k.replace("_", " ").lower(): v.replace("_", " ") for k, v in redirects_en.items()}

        ############ Begin redirect entities ############

        entity_mentions_map_applied_redirects = OrderedDict()

        for (entity, mention_list) in entity_mentions_map:
            ent_underscore = entity.replace(" ", "_")

            mention_list = [
                (mention, count)
                for mention, count in mention_list
                if not (len(entity.split(",")) > 1 and " ".join(mention) in entity.lower().split(",")[1])
            ]

            if ent_underscore in redirects_en:
                redirected_entity = redirects_en[ent_underscore].replace("_", " ")
                if redirected_entity not in entity_mentions_map_applied_redirects:
                    entity_mentions_map_applied_redirects[redirected_entity] = mention_list
                else:
                    entity_mentions_map_applied_redirects[redirected_entity] = merge(
                        mention_list, entity_mentions_map_applied_redirects[redirected_entity]
                    )
                    redirected_entity = None
            else:
                entity_mentions_map_applied_redirects[entity] = mention_list

        self.log("entity_mentions_map {:,}".format(len(entity_mentions_map)))

        ############ Begin filter low count entities and if the mention is in entity, like mention: 'United States' and entity 'Texas, United States' " ############
        self.log("Begin filter mention in entities and low count entities")

        entity_mentions_map_filtered_low_count_implicits = [
            (
                entity,
                [
                    (mention, count / sum([count for (mention, count) in mention_list]))
                    for mention, count in mention_list
                    if count / sum([count for (_, count) in mention_list]) > 0.1
                    and not (len(entity.split(",")) > 1 and " ".join(mention) in entity.lower().split(",")[1])
                    or " ".join(mention) in redirects_en_lc_spaces
                    and redirects_en_lc_spaces[" ".join(mention)] == entity
                ],
            )
            for (entity, mention_list) in tqdm.tqdm(entity_mentions_map_applied_redirects.items())
        ]

        entity_mentions_map_filtered_low_count_implicits_dict = {
            k: OrderedDict(v) for (k, v) in entity_mentions_map_filtered_low_count_implicits
        }

        ############ Create mention_entity_dict  ############
        self.log("Create mention_entity_dict")

        mention_entity_dict = OrderedDict()

        for (entity, mention_list) in tqdm.tqdm(entity_mentions_map_applied_redirects.items()):
            for mention, count in mention_list:
                if (
                    count / sum([count for (_, count) in mention_list]) > 0.1
                    and not (len(entity.split(",")) > 1 and " ".join(mention) in entity.lower().split(",")[1])
                    or " ".join(mention) in redirects_en_lc_spaces
                    and redirects_en_lc_spaces[" ".join(mention)] == entity
                ):
                    if mention not in mention_entity_dict:
                        mention_entity_dict[mention] = Counter()
                    mention_entity_dict[mention][entity] += count

        total_entity_count = 0
        for mention, entities in tqdm.tqdm(mention_entity_dict.items()):
            total_entity_count += sum(entities.values())

        self.log("total_entity_count {:,}".format(total_entity_count))
        self.log("entity_mentions_map_filtered_low_count_implicits {:,}".format(len(entity_mentions_map_filtered_low_count_implicits)))

        with open(f"data/versions/{self.opts.data_version_name}/indexes/mention_entity_dict.pickle", "wb") as f:
            pickle.dump(mention_entity_dict, f)

        with open(f"data/versions/{self.opts.data_version_name}/indexes/entity_mentions_map_filtered_low_count_implicits_dict.pickle", "wb",) as f:
            pickle.dump(entity_mentions_map_filtered_low_count_implicits_dict, f)
