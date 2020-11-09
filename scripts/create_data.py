import os

from preprocessing.create_elasticsearch_index import CreateElasticsSearch
from preprocessing.create_training_data import CreateTrainingData
from preprocessing.map_to_ids import MapToIds
from preprocessing.pipeline_job import PipelineJob
from preprocessing.create_redirects import CreateRedirects
from preprocessing.process_avro import ProcessAVRO
from preprocessing.process_entities_and_mentions import ProcessMentionAndEntities
from preprocessing.process_triples import ProcessTriples
from preprocessing.sample_evaluation_data import SampleEvaluation
from utils.misc import argparse_bool_type
import configargparse as argparse
import logging


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", is_config_file=True, help="config file path")
parser.add_argument("--debug", type=argparse_bool_type, default=False)
parser.add_argument("--verbose", type=argparse_bool_type, default=False)
parser.add_argument("--data_version_name", type=str, default="acl2020", help="data identifier/version")
parser.add_argument("--uncased", type=argparse_bool_type, default=True)

# preprocess avro options
parser.add_argument("--process_avro__log_every", type=int, default=1000)
parser.add_argument("--process_avro__opiec_dir", type=str, default="OPIEC-Clean")
parser.add_argument("--process_avro__debug_file_nr", type=int, default=2)
parser.add_argument("--process_avro__nr_of_workers", type=int, default=6)
parser.add_argument("--process_avro__len_relation_word", type=int, default=10)
parser.add_argument("--process_avro__len_subject_word", type=int, default=10)
parser.add_argument("--process_avro__len_object_word", type=int, default=10)
parser.add_argument("--process_avro__minimum_total_count_for_mentions_per_entity", type=int, default=5)
parser.add_argument("--process_avro__the_top_k_mentions_per_entity", type=int, default=10)
parser.add_argument("--process_avro__min_mention_occurrence_count", type=int, default=5)

# preprocess triples options
parser.add_argument("--process_triples__mention_tokens_vocab_size", type=int, default=20000) # 200009
parser.add_argument("--process_triples__relation_tokens_vocab_size", type=int, default=5000) # 50000

# sample_evaluation options
parser.add_argument("--sample_evaluation__min_relation_token_length_for_testing", type=int, default=3)
parser.add_argument("--sample_evaluation__eval_data_size", type=int, default=50) # 10000

# create_elasticsearch_index options
parser.add_argument("--create_elasticsearch_index__host", type=str, default="localhost")



parser.add_argument("--num_workers", type=int, default="10")

args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

for k, v in args.__dict__.items():
    logging.info(f"{k}: {v}")
    if v == "None":
        args.__dict__[k] = None

os.makedirs(f"data/versions/{args.data_version_name}/", exist_ok=True)

with open(f"data/versions/{args.data_version_name}/config.yaml", "w") as f:
    f.writelines(["{}: {}\n".format(k, v) for k, v in args.__dict__.items()])

PipelineJob.run_jobs([
    ProcessAVRO,
    CreateRedirects,
    ProcessMentionAndEntities,
    ProcessTriples,
    CreateElasticsSearch,
    SampleEvaluation,
    CreateTrainingData,
    MapToIds,
], args)
