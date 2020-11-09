# Can We Predict New Facts with Open Knowledge Graph Embeddings? A Benchmark for Open Link Prediction

This repository contains the code for the ACL 2020 paper [**"Can We Predict New Facts with Open Knowledge Graph Embeddings? A Benchmark for Open Link Prediction"**](https://www.aclweb.org/anthology/2020.acl-main.209/). The code is provided as a documentation for the paper and also for follow-up research.

# <p align="center"> <img src="docs/lp_vs_olp.png" alt="link prediction vs open link prediction" width="70%"> </p>

The content of this page covers the following topics: 

1. [Preparation and Installation](#preparation-and-installation)
2. [Training Open Knowledge Graph Embedding Model on OLPBENCH](#training)
3. [Issues and possible improvements](#issues-and-possible-improvements)
















## Preparation and Installation 

- The project is installed as follows:

    ```
    git clone https://github.com/samuelbroscheit/open_link_prediction_benchmark.git
    cd open_link_prediction_benchmark
    pip install -r requirements.txt
    ```

- Add paths to environment

    ```
    source setup_paths
    ```

- Download OLPBENCH

    Download the full dataset (compressed: ~ 2.4 GB, uncompressed: ~ 7.9 GB)

    ```
    cd data
    wget http://data.dws.informatik.uni-mannheim.de/olpbench/olpbench.tar.gz
    tar xzf olpbench.tar.gz
    cd ..
    ```
- Download OPIEC

    **Only** if you want to recreate OLPBENCH from scratch!

    Download the OPIEC clean dataset (compressed: ~ 35 GB, uncompressed: ~ 292.4 GB)

    ```
    cd data
    wget http://data.dws.informatik.uni-mannheim.de/opiec/OPIEC-Clean.zip
    unzip OPIEC-Clean.zip
    cd ..
    ```

    Then download and start and Elasticsearch server, that should listen on localhost:9200 . This is usually as easy as downloading the most recent version, unzip it, change the default configuration to

    ```
    cluster node.local: true # disable network
    ```

    and then start the server in with ./bin/elasticsearch. Then run the preprocessing with 
    
    ```
    python scripts/create_data.py -c config/preprocessing/prototype.yaml
    ```


###### Prepared configurations
    

- [config/preprocessing/prototype.yaml](config/preprocessing/prototype.yaml) a configuration for prototyping

- [config/preprocessing/acl2020.yaml](config/preprocessing/acl2020.yaml) the configurations with the settings fomr the ACL2020 study









## Training

Once preparation and installation are finished you can train a model on OLPBENCH. 

1. [Run training](#run-training)
2. [Prepared configurations](#prepared-configurations)
3. [Available options](#available-options)

### Run training

Run the training with:

```  
python scripts/train.py [TRAIN_CONFIG_YAML] [OPTIONS] 
```  

TRAIN_CONFIG_YAML is a yaml config file. The possible options are documented in:

[openkge/default.yaml](openkge/default.yaml)

All top level options can also be set on the command line and override the yaml confguration.


###### Prepared configurations
    
A token-based model for the OLPBench benchmark.

- [config/acl2020-openlink/wikiopenlink-thorough-complex-lstm.yaml](config/acl2020-openlink/wikiopenlink-thorough-complex-lstm.yaml) a configuration for one of the experiments of the ACL2020 study.


A various models for the the Freebase FB15k-237 benchmark. Suggested for prototyping token based models.

- [config/fb15k237/fb15k237-complex-lstm.yaml](config/fb15k237/fb15k237-complex-lstm.yaml)  

- [config/fb15k237/fb15k237-complex-unigrampool.yaml](config/fb15k237/fb15k237-complex-unigrampool.yaml) 

- [config/fb15k237/fb15k237-complex-kge.yaml](config/fb15k237/fb15k237-complex-kge.yaml) 

### Run evaluation

Run evaluation on test data with:

```  
python scripts/train.py --resume data/experiments/.../checkpoint.pth.tar --evaluate True --evaluate_on_validation False
```  

_--resume_ epects the path to a checkpoint file

_--evaluate_on_validation False_ sets the evaluation to run on test data


### Prepared configurations

In the config folder you will find the following configurations:

- [config/acl2020-openlink/wikiopenlink-thorough-complex-lstm.yaml](config/acl2020-openlink/wikiopenlink-thorough-complex-lstm.yaml) is a configuration to train a OpenKGE model on the open link benchmark data.

### Models

###### Lookup based models (standard KGE)

-    LookupTucker3RelationModel 
-    LookupDistmultRelationModel
-    LookupComplexRelationModel

###### Token based models 

*Compute the entity and relation embeddings by pooling token embeddings*
-    UnigramPoolingComplexRelationModel

*Compute the entity and relation embeddings with a sliding window CNN*
-    BigramPoolingComplexRelationModel 

*Compute the entity and relation embeddings with a LSTM*
-    LSTMDistmultRelationModel 
-    LSTMComplexRelationModel
-    LSTMTucker3RelationModel

###### Diagnostic models

-    DataBiasOnlyEntityModel 
-    DataBiasOnlyRelationModel 


For model options see the init of the respective class. Additional combinations of score and embedding functions can be easily created:

```  
class BigramPoolingDistmultRelationModel(DistmultRelationScorer, BigramPoolingRelationEmbedder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
```  




## Citation

if you find this code useful for your research please cite 

```
@inproceedings{broscheit-etal-2020-predict,
    title = "Can We Predict New Facts with Open Knowledge Graph Embeddings? A Benchmark for Open Link Prediction",
    author = "Broscheit, Samuel  and
      Gashteovski, Kiril  and
      Wang, Yanjie  and
      Gemulla, Rainer",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.209",
    doi = "10.18653/v1/2020.acl-main.209",
    pages = "2296--2308",
}
```
