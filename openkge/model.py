import logging
from typing import Union

import torch
import torch.nn.functional

from tqdm import tqdm

from openkge.dataset import EntityRelationDatasetMeta, OneToNMentionRelationDataset
from openkge.index_mapper import PAD
from utils.torch_nn_modules import Sequential


class RelationModel(torch.nn.Module):
    r"""Base class for all relational models.
    """

    is_cuda = False
    rel_obj_cache = None
    subj_rel_cache = None

    def cuda(self, device=None):
        super().cuda(device=device)
        self.is_cuda = True

    def cpu(self):
        super().cpu()
        self.is_cuda = False


class RelationScorer(RelationModel):
    r"""Base class for all relational model scorer.
    """

    def forward(self, subj, rel, obj, **kwargs):

        subj = self.encode_subj(subj)
        rel = self.encode_rel(rel)
        obj = self.encode_obj(obj)

        return self.triple_score(subj, rel, obj, **kwargs)

    def triple_score(self, subj, rel, obj, **kwargs):
        """    Computes the score during training. This might differ from
        the computation during prediction because of efficiency reasons
        when we predict over all entities.
        Has to be implemented.
        """
        raise NotImplementedError


    def sp_prefix_score(self, subj=None, rel=None, many_obj=None):
        """    Computes the score during training. This might differ from
        the computation during prediction because of efficiency reasons
        when we predict over all entities.
        Has to be implemented.
        """
        subj = self.encode_subj(subj)
        rel = self.encode_rel(rel)
        if many_obj is None:
            many_obj = self.get_all_obj()
        return self._score(subj, rel, many_obj, prefix=True, sp=True, po=False)

    def po_prefix_score(self, rel=None, obj=None, many_subj=None):
        """    Computes the score during training. This might differ from
        the computation during prediction because of efficiency reasons
        when we predict over all entities.
        Has to be implemented.
        """
        if many_subj is None:
            many_subj = self.get_all_subj()
        rel = self.encode_rel(rel)
        obj = self.encode_obj(obj)
        return self._score(many_subj, rel, obj, prefix=True, sp=False, po=True)

    def precompute_batch_shared_inputs(self, entity_ids):
        return self.encode_obj(entity_ids)


class RelationEmbedder(RelationModel):
    r"""Base class for all relational model embedder.
    """

    def encode_subj(self, subj) -> torch.Tensor:
        """  Computes the embedding for subject tokens.
        Can be one token for simple lookup or a sequence of tokens.
        """
        raise NotImplementedError

    def encode_rel(self, rel) -> torch.Tensor:
        """  Computes the embedding for relation tokens.
        Can be one token for simple lookup or a sequence of tokens.
        """
        raise NotImplementedError

    def encode_obj(self, obj) -> torch.Tensor:
        """  Computes the embedding for object tokens.
        Can be one token for simple lookup or a sequence of tokens.
        """
        raise NotImplementedError

    def get_all_subj(self) -> torch.Tensor:
        """  Returns all subject embeddings.
        Might trigger a precomputation for embedders which handle sequence of tokens.
        """
        raise NotImplementedError

    def get_all_rel(self) -> torch.Tensor:
        """  Returns all relation embeddings.
        Might trigger a precomputation for embedders which handle sequence of tokens.
        """
        raise NotImplementedError

    def get_all_obj(self) -> torch.Tensor:
        """  Returns all object embeddings.
        Might trigger a precomputation for embedders which handle sequence of tokens.
        """
        raise NotImplementedError

    def get_subj(self, subj) -> torch.Tensor:
        """  Returns one subject embedding.
        Might trigger a precomputation for embedders which handle sequence of tokens.
        """
        raise NotImplementedError

    def get_rel(self, rel) -> torch.Tensor:
        """  Returns one relation embedding.
        Might trigger a precomputation for embedders which handle sequence of tokens.
        """
        raise NotImplementedError

    def get_obj(self, obj) -> torch.Tensor:
        """  Returns one object embedding.
        Might trigger a precomputation for embedders which handle sequence of tokens.
        """
        raise NotImplementedError

    def precompute_embeddings_from_tokens(self):
        raise NotImplementedError


class RescalRelationScorer(RelationScorer):

    def triple_score(self, subj, rel, obj, drop_relation=False):
        return self._score(subj, rel, obj,)

    def _score(self, subj, rel, obj, prefix=False, drop_relation=False, sp=None, po=None):
        r"""
        :param subj: tensor of size [batch_sz, embedding_size]
        :param rel: tensor of size [batch_sz, embedding_size*embedding_size]
        :param obj: tensor of size [batch_sz, embedding_size]
        :return: score tensor of size [batch_sz, 1]
        """

        batch_sz = rel.size(0)
        subj = subj.view(-1, subj.size(-1))
        rel = rel.view(batch_sz, subj.size(-1), subj.size(-1))
        obj = obj.view(-1, obj.size(-1))

        if prefix:
            if sp:
                out = (subj.view(-1, 1, subj.size(-1)).bmm(rel)).view(-1, subj.size(-1)).mm(obj.transpose(0,1))
            elif po:
                out = rel.bmm(obj.view(-1, obj.size(-1), 1)).view(-1, obj.size(-1)).mm(subj.transpose(0,1))
            else:
                raise Exception
        else:
            subj = subj.view(batch_sz, 1, self.slot_size)
            rel = rel.view(batch_sz, self.slot_size, self.slot_size)
            obj = obj.view(batch_sz, self.slot_size, 1)
            out = (subj.bmm(rel.bmm(obj)))

        return out.view(batch_sz, -1)


class ComplexRelationScorer(RelationScorer):

    def triple_score(self, subj, rel, obj, drop_relation=False):
        return self._score(subj, rel, obj,)

    def _score(self, subj, rel, obj, prefix=False, sp=None, po=None):
        r"""
        :param subj: tensor of size [batch_sz, embedding_size]
        :param rel: tensor of size [batch_sz, embedding_size]
        :param obj: tensor of size [batch_sz, embedding_size]
        :return: score tensor of size [batch_sz, 1]

        """

        batch_sz = rel.size(0)

        subj = subj.view(-1, subj.size(-1))
        rel = rel.view(-1, rel.size(-1))
        obj = obj.view(-1, obj.size(-1))

        feat_dim = 1

        if prefix:

            def _compute_score(subj, rel, obj, pr=''):
                # if len(pr)>0: print(pr, subj.size(0), rel.size(0), obj.size(0))
                rel1, rel2 = (t.contiguous() for t in rel.chunk(2, dim=feat_dim))
                subj1, subj2 = (t.contiguous() for t in subj.chunk(2, dim=feat_dim))
                obj1, obj2 = (t.contiguous() for t in obj.chunk(2, dim=feat_dim))
                if sp:
                    out = (subj1 * rel1).mm(obj1.transpose(0, 1)) + \
                          (subj2 * rel1).mm(obj2.transpose(0, 1)) + \
                          (subj1 * rel2).mm(obj2.transpose(0, 1)) - \
                          (subj2 * rel2).mm(obj1.transpose(0, 1))
                    return out
                elif po:
                    out = (obj1 * rel1).mm(subj1.transpose(0, 1)) + \
                          (obj2 * rel1).mm(subj2.transpose(0, 1)) + \
                          (obj2 * rel2).mm(subj1.transpose(0, 1)) - \
                          (obj1 * rel2).mm(subj2.transpose(0, 1))
                    return out
                else:
                    raise Exception

            # chunk entity prediction
            # tmp_batch_sz = 1024 * 8
            tmp_batch_sz = 1024 * 16
            # print("PRE", subj.size(0), rel.size(0), obj.size(0))
            if obj.size(0) > tmp_batch_sz:
                out = torch.cat([_compute_score(subj, rel, _obj, pr='OBJ') for _obj in obj.chunk(obj.size(0) // tmp_batch_sz + 1)], dim=1)
            elif subj.size(0) > tmp_batch_sz:
                out = torch.cat([_compute_score(_subj, rel, obj, pr='SUBJ') for _subj in subj.chunk(subj.size(0) // tmp_batch_sz + 1)], dim=1)
            else:
                out = _compute_score(subj, rel, obj)

        else:
            rel1, rel2 = (t.contiguous() for t in rel.chunk(2, dim=feat_dim))
            obj1, obj2 = (t.contiguous() for t in obj.chunk(2, dim=feat_dim))
            subj_all = torch.cat((subj, subj), dim=feat_dim)
            rel_all = torch.cat((rel1, rel, -rel2,), dim=feat_dim)
            obj_all = torch.cat((obj, obj2, obj1,), dim=feat_dim)

            out = (subj_all * obj_all * rel_all).sum(dim=feat_dim)

        return out.view(batch_sz, -1)


class DistmultRelationScorer(RelationScorer):

    def triple_score(self, subj, rel, obj, drop_relation=False):
        return self._score(subj, rel, obj,)

    def _score(self, subj, rel, obj, prefix=False, drop_relation=False, sp=None, po=None):
        r"""
        :param subj: tensor of size [batch_sz, embedding_size]
        :param rel: tensor of size [batch_sz, embedding_size]
        :param obj: tensor of size [batch_sz, embedding_size]
        :return: score tensor of size [batch_sz, 1]

        Because the backward is more expensive for n^2 we use the Hadamard form during training

        """
        # if not self.training:
        #     print(subj.size(), rel.size(), obj.size())
        batch_sz = rel.size(0)

        subj = subj.view(-1, subj.size(-1))
        rel = rel.view(-1, rel.size(-1))
        obj = obj.view(-1, obj.size(-1))

        feat_dim = 1

        if prefix:
            if sp:
                out = (subj * rel).mm(obj.transpose(0,1))
            elif po:
                out = (rel * obj).mm(subj.transpose(0,1))
            else:
                raise Exception
        else:
            out = (subj * obj * rel).sum(dim=feat_dim)

        return out.view(batch_sz, -1)


class DataBiasOnlyRelationScorer(RelationScorer):

    def triple_score(self, subj, rel, obj, drop_relation=False):
        return self._score(subj, rel, obj,)

    def _score(self, subj, rel, obj, prefix=False, drop_relation=False, sp=None, po=None):
        r"""
        :param subj: tensor of size [batch_sz, embedding_size]
        :param rel: tensor of size [batch_sz, embedding_size]
        :param obj: tensor of size [batch_sz, embedding_size]
        :return: score tensor of size [batch_sz, 1]

        """
        # if not self.training:
        #     print(subj.size(), rel.size(), obj.size())
        batch_sz = rel.size(0)

        subj = subj.view(-1, subj.size(-1))
        rel = rel.view(-1, rel.size(-1))
        obj = obj.view(-1, obj.size(-1))

        feat_dim = 1

        if prefix:
            if sp:
                out = rel.mm(obj.transpose(0,1))
            elif po:
                out = rel.mm(subj.transpose(0,1))
            else:
                raise Exception
        else:
            raise Exception

        return out.view(batch_sz, -1)


class DataBiasOnlyEntityScorer(RelationScorer):

    def triple_score(self, subj, rel, obj, drop_relation=False):
        return self._score(subj, rel, obj,)

    def _score(self, subj, rel, obj, prefix=False, drop_relation=False, sp=None, po=None):
        r"""
        :param subj: tensor of size [batch_sz, embedding_size]
        :param rel: tensor of size [batch_sz, embedding_size]
        :param obj: tensor of size [batch_sz, embedding_size]
        :return: score tensor of size [batch_sz, 1]

        """
        # if not self.training:
        #     print(subj.size(), rel.size(), obj.size())
        batch_sz = rel.size(0)

        subj = subj.view(-1, subj.size(-1))
        rel = rel.view(-1, rel.size(-1))
        obj = obj.view(-1, obj.size(-1))

        feat_dim = 1

        if prefix:
            if sp:
                out = subj.mm(obj.transpose(0,1))
            elif po:
                out = obj.mm(subj.transpose(0,1))
            else:
                raise Exception
        else:
            raise Exception

        return out.view(batch_sz, -1)


class LookupBaseRelationEmbedder(RelationEmbedder):

    def __init__(self,
                 entity_slot_size,
                 relation_slot_size,
                 train_data:EntityRelationDatasetMeta,
                 entity_embedding_size=None,
                 relation_embedding_size=None,
                 normalize='',
                 dropout=0.0,
                 input_dropout=0.0,
                 relation_dropout=0.0,
                 relation_input_dropout=0.0,
                 project_entity=False,
                 project_entity_activation='ReLU',
                 project_relation=True,
                 project_relation_activation=None,
                 sparse=False,
                 init_std=0.01,
                 batch_norm=False,
                 l2_reg=0,
                 ):
        super().__init__()

        self.train_data=train_data

        if relation_slot_size is None or relation_slot_size <= 0:
            relation_slot_size = entity_slot_size

        self._entity_embedding_size = entity_embedding_size
        if entity_embedding_size is None:
            self._entity_embedding_size = entity_slot_size

        self._relation_embedding_size = relation_embedding_size
        if relation_embedding_size is None:
            self._relation_embedding_size = relation_slot_size

        self.entity_embedding = torch.nn.Embedding(train_data.entities_size, self._entity_embedding_size, sparse=sparse, padding_idx=PAD)
        self.relation_embedding = torch.nn.Embedding(train_data.relations_size, self._relation_embedding_size, sparse=sparse, padding_idx=PAD)

        if hasattr(train_data, "entity_id_sparse_rescaler_map"):
            self.entity_sparse_rescaler_lookup = torch.nn.Embedding(train_data.entities_size, 1, sparse=sparse, )
            self.entity_sparse_rescaler_lookup.weight.data = torch.FloatTensor([v for k,v in sorted(train_data.entity_id_sparse_rescaler_map.items(), key=lambda x:x[0])]).view(-1, 1)
            self.entity_sparse_rescaler_lookup.weight.requires_grad = False
            self.relation_sparse_rescaler_lookup = torch.nn.Embedding(train_data.relations_size, 1, sparse=sparse,)
            self.relation_sparse_rescaler_lookup.weight.data = torch.FloatTensor([v for k,v in sorted(train_data.relation_id_sparse_rescaler_map.items(), key=lambda x:x[0])]).view(-1, 1)
            self.relation_sparse_rescaler_lookup.weight.requires_grad = False

        # Projection for relation / core tensor
        if project_relation:
            project_relation_activation_class = None
            if project_relation_activation:
                project_relation_activation_class = getattr(torch.nn, project_relation_activation)()
            project_relation_layer = torch.nn.Linear(self._relation_embedding_size, entity_slot_size ** 2, bias=False)
            torch.nn.init.xavier_normal_(project_relation_layer.weight.data)
            self.relation_projection = Sequential(project_relation_layer, project_relation_activation_class)

        if project_entity:
            project_entity_activation_class = None
            if project_entity_activation:
                project_entity_activation_class = getattr(torch.nn, project_entity_activation)()
            project_subject_layer = torch.nn.Linear(entity_slot_size, entity_slot_size, bias=False)
            project_object_layer = torch.nn.Linear(entity_slot_size, entity_slot_size, bias=False)
            torch.nn.init.xavier_normal_(project_subject_layer.weight.data)
            torch.nn.init.xavier_normal_(project_object_layer.weight.data)
            self.subj_projection = Sequential(project_subject_layer, project_entity_activation_class)
            self.obj_projection = Sequential(project_object_layer, project_entity_activation_class)

        self.project_entity = project_entity
        self.project_relation = project_relation
        self.slot_size = entity_slot_size
        self.rel_obj_cache = None
        self.subj_rel_cache = None
        self.normalize = normalize

        # Initialize parameters
        torch.nn.init.normal_(self.entity_embedding.weight.data, std=init_std)
        torch.nn.init.normal_(self.relation_embedding.weight.data, std=init_std)

        self.dropout = dropout
        self.input_dropout = input_dropout
        self.relation_dropout = dropout if relation_dropout is None else relation_dropout
        self.relation_input_dropout = input_dropout if relation_input_dropout is None else relation_input_dropout

        # self.register_buffer('eye', torch.eye(self.relation_embedding.weight.size(0),self.relation_embedding.weight.size(0)), )

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn_e = torch.nn.BatchNorm1d(self._entity_embedding_size)
            self.bn_r = torch.nn.BatchNorm1d(self._relation_embedding_size)

        self.l2_reg = l2_reg
        self._l2_reg_hook = None

    def after_batch_loss_hook(self, epoch):
        if self.training:
            if self.l2_reg > 0:
                result = self._l2_reg_hook
                self._l2_reg_hook = None
                return result
        return None

    def _encode(self, slot_item, embedding, project, input_dropout, dropout, batch_norm=None, lookup=True):
        if lookup:
            slot_item = slot_item.squeeze()
            repr = embedding(slot_item.long())
        else:
            repr = slot_item
        if input_dropout > 0:
            repr = torch.nn.functional.dropout(repr, p=input_dropout, training=self.training)
        if self.batch_norm:
            repr = batch_norm(repr)
        if project:
            repr = project(repr)
        if self.normalize == 'norm':
            repr = torch.nn.functional.normalize(repr)
        if dropout > 0:
            repr = torch.nn.functional.dropout(repr, p=dropout, training=self.training)
        if self.training and self.l2_reg > 0:
            _l2_reg_hook = repr
            if self.dropout > 0:
                _l2_reg_hook =_l2_reg_hook / self.dropout
            _l2_reg_hook = self.l2_reg*_l2_reg_hook.abs().pow(3).sum()
            if self._l2_reg_hook is None:
                self._l2_reg_hook = _l2_reg_hook
            else:
                self._l2_reg_hook = self._l2_reg_hook + _l2_reg_hook
        return repr

    def encode_rel(self, rel, lookup=True):
        return self._encode(rel,
                            self.relation_embedding,
                            self.relation_projection if self.project_relation else None,
                            self.relation_input_dropout,
                            self.relation_dropout,
                            self.bn_r if self.batch_norm else None,
                            lookup=lookup
                            )

    def encode_subj(self, subj, lookup=True):
        return self._encode(subj,
                            self.entity_embedding,
                            self.subj_projection if self.project_entity else None,
                            self.input_dropout,
                            self.dropout,
                            self.bn_e if self.batch_norm else None,
                            lookup=lookup
                            )

    def encode_obj(self, obj, lookup=True):
        return self._encode(obj,
                            self.entity_embedding,
                            self.obj_projection if self.project_entity else None,
                            self.input_dropout,
                            self.dropout,
                            self.bn_e if self.batch_norm else None,
                            lookup=lookup
                            )

    def _get_all(self, min_size, encode_func, embedding):
        result = encode_func(embedding.weight[min_size:].contiguous(), lookup=False)
        return result

    def get_all_rel(self):
        return self._get_all(self.train_data.min_relations_size, self.encode_rel, self.relation_embedding)

    def get_all_subj(self):
        return self._get_all(self.train_data.min_entities_size, self.encode_subj, self.entity_embedding)

    def get_all_obj(self):
        return self._get_all(self.train_data.min_entities_size, self.encode_obj, self.entity_embedding)

    def _get(self, encode_func, id):
        id = torch.IntTensor([id]).pin_memory()
        if self.is_cuda:
            id = id.cuda()
        result = encode_func(id)
        return result

    def get_subj(self, subj):
        return self._get(self.encode_subj, subj)

    def get_rel(self, rel):
        return self._get(self.encode_rel, rel)

    def get_obj(self, obj):
        return self._get(self.encode_obj, obj)

    def get_slot_size(self):
        return self.slot_size


class LookupSimpleRelationEmbedder(LookupBaseRelationEmbedder):

    def __init__(self,
                 entity_slot_size,
                 **kwargs,
                 ):
        if 'relation_slot_size' in kwargs: del kwargs['relation_slot_size']
        super().__init__(
            entity_slot_size=entity_slot_size,
            relation_slot_size=entity_slot_size,
            project_relation=False,
            **kwargs,
        )
        self.relation_projection = None


class TokenBasedRelationEmbedder(RelationEmbedder):

    def __init__(self,
                 train_data: EntityRelationDatasetMeta,
                 entity_slot_size: int,
                 relation_slot_size: int,
                 sparse: bool,
                 init_std: float,
                 normalize=None,
                 ):
        super().__init__()
        if relation_slot_size is None or relation_slot_size <= 0:
            relation_slot_size = entity_slot_size
        self.train_data=train_data

        entity_max_len = self.train_data.max_length[0]
        relation_max_len = self.train_data.max_length[1]

        entity_token_ids = torch.zeros(len(self.train_data.entity_id_to_tokens_map), entity_max_len).long()
        for entity_id, seq in enumerate(self.train_data.entity_id_to_tokens_map):
            entity_token_ids[entity_id].narrow(0, 0, min(entity_max_len, len(
                self.train_data.entity_id_to_tokens_map[entity_id]))).copy_(
                torch.IntTensor(
                    self.train_data.entity_id_to_tokens_map[entity_id][
                    -entity_max_len:]))
        self.register_buffer('entity_token_ids', entity_token_ids)

        relation_token_ids = torch.zeros(len(self.train_data.relation_id_to_tokens_map), relation_max_len).long()
        for relation_id, seq in enumerate(self.train_data.relation_id_to_tokens_map):
            relation_token_ids[relation_id].narrow(0, 0, min(relation_max_len, len(
                self.train_data.relation_id_to_tokens_map[relation_id]))).copy_(
                torch.IntTensor(
                    self.train_data.relation_id_to_tokens_map[relation_id][
                    -relation_max_len:]))
        self.register_buffer('relation_token_ids', relation_token_ids)

        self.entity_embedding = torch.nn.Embedding(
            train_data.entity_tokens_size,
            entity_slot_size,
            sparse=sparse,
            padding_idx=0
        )
        self.relation_embedding = torch.nn.Embedding(
            train_data.relation_tokens_size,
            relation_slot_size,
            sparse=sparse,
            padding_idx=0
        )

        self.entity_batchnorm = None
        self.relation_batchnorm = None
        self.normalize = normalize
        if normalize == 'batchnorm':
            self.entity_batchnorm = torch.nn.BatchNorm1d(entity_slot_size, momentum=0.1, eps=1e-5)
            self.relation_batchnorm = torch.nn.BatchNorm1d(relation_slot_size, momentum=0.1, eps=1e-5)
            torch.nn.init.uniform_(self.entity_batchnorm.weight)
            torch.nn.init.uniform_(self.relation_batchnorm.weight)

        self.entity_to_tokens_map = train_data.entity_id_to_tokens_map
        self.relation_to_tokens_map = train_data.relation_id_to_tokens_map
        self.num_entity_words = train_data.entity_tokens_size
        self.train_data.relations_size_words = train_data.relation_tokens_size
        self.precompute_entity_embedding_from_tokens = True
        self.entity_embedding_from_tokens = None
        self.precompute_relations_embedding_from_tokens = True
        self.relations_embedding_from_tokens = None
        self.train_data.entities_size = train_data.entities_size
        self.train_data.relations_size = train_data.relations_size
        self.rel_obj_cache = None
        self.subj_rel_cache = None
        self.slot_size = entity_slot_size
        self.relation_slot_size = relation_slot_size
        torch.nn.init.normal_(self.entity_embedding.weight.data, std=init_std)
        torch.nn.init.normal_(self.relation_embedding.weight.data, std=init_std)

    def get_all_subj(self):
        self.precompute_embeddings_from_tokens()
        return self.entity_embedding_from_tokens[self.train_data.min_entities_size:]

    def get_all_rel(self):
        self.precompute_embeddings_from_tokens()
        return self.relations_embedding_from_tokens[self.train_data.min_entities_size:]

    def get_all_obj(self):
        self.precompute_embeddings_from_tokens()
        return self.entity_embedding_from_tokens[self.train_data.min_entities_size:]

    def get_subj(self, subj):
        self.precompute_embeddings_from_tokens()
        return self.entity_embedding_from_tokens[subj].unsqueeze(0)

    def get_rel(self, rel):
        self.precompute_embeddings_from_tokens()
        return self.relations_embedding_from_tokens[rel]

    def get_obj(self, obj):
        self.precompute_embeddings_from_tokens()
        return self.entity_embedding_from_tokens[obj].unsqueeze(0)

    def eval(self, *args, **kwargs):
        self.entity_embedding_from_tokens = None
        self.relation_embedding_from_tokens = None
        super(TokenBasedRelationEmbedder, self).eval()

    def train(self, *args, **kwargs):
        self.entity_embedding_from_tokens = None
        self.relation_embedding_from_tokens = None
        super(TokenBasedRelationEmbedder, self).train(*args, **kwargs)

    def precompute_embeddings_from_tokens(self):
        if self.entity_embedding_from_tokens is None:
            entity_size = self.train_data.entities_size
            relation_size = self.train_data.relations_size
            logging.debug("Precompute embeddings from tokens, "
                         "{} entity embeddings and "
                         "{} relation embeddings ... ".format(
                entity_size,
                relation_size
            )
            )
            self.eval()
            with torch.no_grad():
                batch_size = 1024 * 4
                if self.precompute_entity_embedding_from_tokens:
                    if not hasattr(self, 'encode_entities_from_tokens') or hasattr(self, 'encode_entities_from_tokens') and self.encode_entities_from_tokens:

                        self.entity_embedding_from_tokens = torch.zeros(torch.Size((entity_size, self.entity_embedding.weight.size(-1))))
                        eids = torch.IntTensor(list(range(entity_size)))
                        if self.is_cuda:
                            eids = eids.cuda()
                        for begin in tqdm(range(0, entity_size, batch_size)):
                            end = min(begin+batch_size, entity_size)
                            self.entity_embedding_from_tokens[begin:end] = self.encode_subj(eids[begin:end]).view(end-begin, -1)

                        if self.is_cuda:
                            self.entity_embedding_from_tokens = self.entity_embedding_from_tokens.cuda()

                    logging.debug(" .... {} entities finished .... ".format(self.entity_embedding_from_tokens.size()))

                if self.precompute_relations_embedding_from_tokens:
                    if not hasattr(self, 'encode_relations_from_tokens') or hasattr(self, 'encode_relations_from_tokens') and self.encode_relations_from_tokens:
                        self.relations_embedding_from_tokens = torch.zeros(torch.Size((relation_size, self.relation_slot_size)))
                        rids = torch.IntTensor(list(range(relation_size)))
                        if self.is_cuda:
                            rids = rids.cuda()
                        for begin in tqdm(range(0, relation_size, batch_size)):
                            end = min(begin+batch_size, relation_size)
                            self.relations_embedding_from_tokens[begin:end] = self.encode_rel(rids[begin:end]).view(end-begin, -1)

                        if self.is_cuda:
                            self.relations_embedding_from_tokens = self.relations_embedding_from_tokens.cuda()
                    logging.debug(" .... {} relations finished.".format(self.relations_embedding_from_tokens.size()))



class UnigramPoolingRelationEmbedder(TokenBasedRelationEmbedder):

    def __init__(self,
                 entity_slot_size,
                 relation_slot_size,
                 train_data: EntityRelationDatasetMeta,
                 pool='sum',
                 normalize=None,
                 dropout=0.0,
                 entity_dropout=None,
                 relation_dropout=None,
                 sparse=False,
                 init_std=0.01,
                 activation=None,
                 project_relation=False,
                 ):
        super().__init__(
            entity_slot_size=entity_slot_size,
            relation_slot_size=relation_slot_size,
            train_data=train_data,
            sparse=sparse,
            init_std=init_std,
            normalize=normalize,
        )
        if relation_slot_size is None or relation_slot_size <= 0:
            relation_slot_size = entity_slot_size
        self.relation_slot_size = relation_slot_size
        self.relation_projection = None
        if project_relation:
            self.relation_slot_size = entity_slot_size ** 2
            relation_projection = torch.nn.Linear(relation_slot_size, entity_slot_size ** 2, bias=False)
            init_core_tensor_std = 1 / (entity_slot_size ** 2 * relation_slot_size * init_std ** 3)
            torch.nn.init.normal_(relation_projection.weight.data, init_core_tensor_std)
            self.relation_projection = Sequential(
                relation_projection,
                torch.nn.BatchNorm1d(entity_slot_size ** 2)
            )
        self.normalize = normalize
        self.pool = pool
        self.entity_dropout = entity_dropout if entity_dropout else dropout
        self.relation_dropout = relation_dropout if relation_dropout else dropout

        self.activation = None
        if activation is not None and hasattr(torch.nn, activation):
                self.activation = getattr(torch.nn, activation)()

    def _map_to_tokens(self, input, mapper):
        return torch.nn.functional.embedding(input.long(), mapper, 0, None, 0., False, True).view(input.size(0), -1)

    def _encode(self, input, embedder, token_ids, proj, dropout, norm_func):
        input = self._map_to_tokens(input, token_ids)
        embedded = embedder(input.long())
        if self.pool == 'max':
            encoded, _ = embedded.max(dim=1)
        elif self.pool == 'mean':
            lengths = (input > 0).float().sum(1, keepdim=True)
            encoded = embedded.sum(dim=1)/(lengths+1e-12)
        else:
            encoded = embedded.sum(dim=1)
        if self.activation is not None:
            encoded = self.activation(encoded)
        if self.normalize == 'norm':
            encoded = torch.nn.functional.normalize(encoded, dim=1)
        if self.normalize == 'batchnorm':
            encoded = norm_func(encoded.contiguous())
        if proj:
            encoded = proj(encoded)
        if dropout > 0:
            return torch.nn.functional.dropout(encoded, p=dropout, training=self.training).unsqueeze(1)
        else:
            return encoded.unsqueeze(1)

    def encode_subj(self, subj):
        return self._encode(subj, self.entity_embedding, self.entity_token_ids, self.entity_projection, self.entity_dropout, self.entity_batchnorm)

    def encode_obj(self, obj):
        return self._encode(obj, self.entity_embedding, self.entity_token_ids, self.entity_projection, self.entity_dropout, self.entity_batchnorm)

    def encode_rel(self, rel):
        return self._encode(rel, self.relation_embedding, self.relation_token_ids, self.relation_projection, self.relation_dropout, self.relation_batchnorm)

    def get_slot_size(self):
        return self.slot_size


class BigramPoolingRelationEmbedder(TokenBasedRelationEmbedder):

    def __init__(self,
                 entity_slot_size,
                 relation_slot_size,
                 train_data: EntityRelationDatasetMeta,
                 normalize='',
                 pool='',
                 dropout=None,
                 entity_dropout=None,
                 relation_dropout=None,
                 encoder_activiation=None,
                 sparse=False,
                 init_std=0.01,
                 gates=False,
                 project_relation=False,
                 ):
        super().__init__(
            entity_slot_size=entity_slot_size,
            relation_slot_size=relation_slot_size,
            train_data=train_data,
            sparse=sparse,
            init_std=init_std,
            normalize=normalize,
        )
        if relation_slot_size is None or relation_slot_size <= 0:
            relation_slot_size = entity_slot_size
        self.relation_slot_size = relation_slot_size
        self.relation_projection = None
        if project_relation:
            self.relation_slot_size = entity_slot_size ** 2
            relation_projection = torch.nn.Linear(relation_slot_size, entity_slot_size ** 2, bias=False)
            init_core_tensor_std = 1 / (entity_slot_size ** 2 * relation_slot_size * init_std ** 3)
            torch.nn.init.normal_(relation_projection.weight.data, init_core_tensor_std)
            self.relation_projection = Sequential(
                relation_projection,
                torch.nn.BatchNorm1d(entity_slot_size ** 2)
            )

        self.encoder_activiation = None
        if encoder_activiation is not None and hasattr(torch.nn, encoder_activiation):
            self.encoder_activiation = getattr(torch.nn, encoder_activiation)

        self.normalize = normalize
        self.entity_dropout = entity_dropout if entity_dropout else dropout
        self.relation_dropout = relation_dropout if relation_dropout else dropout
        self.pool = pool
        self.gates = gates

        if encoder_activiation is not None:
            if hasattr(torch.nn, encoder_activiation):
                encoder_activiation = getattr(torch.nn, encoder_activiation)
            else:
                encoder_activiation = None

        self.entity_batchnorm = None
        self.relation_batchnorm = None
        if normalize == 'batchnorm':
            self.entity_batchnorm = torch.nn.BatchNorm1d(entity_slot_size + 1 if self.gates else entity_slot_size, momentum=None)
            self.relation_batchnorm = torch.nn.BatchNorm1d(relation_slot_size + 1 if self.gates else relation_slot_size, momentum=None)

        self.entity_encoder_in = Sequential(
            torch.nn.Conv1d(in_channels=entity_slot_size, out_channels=entity_slot_size + 1 if self.gates else entity_slot_size, kernel_size=2, dilation=1, bias=False),
            encoder_activiation() if encoder_activiation is not None else None,
            self.entity_batchnorm if self.entity_batchnorm is not None else None,
        )

        self.relation_encoder_in = Sequential(
            torch.nn.Conv1d(in_channels=relation_slot_size, out_channels=relation_slot_size + 1 if self.gates else relation_slot_size, kernel_size=2, dilation=1, bias=False),
            encoder_activiation() if encoder_activiation is not None else None,
            self.relation_batchnorm if self.relation_batchnorm is not None else None,
        )

    def _encode(self, input, embedder, encoder, proj, dropout,):
        mask = (input > 0).unsqueeze(1).float()[:, :, 1:]
        embedded = embedder(input).transpose(1, 2)
        encoded = encoder(embedded)
        if self.gates:
            gates = torch.nn.functional.sigmoid(encoded[:, -1, :]).unsqueeze(1)
            encoded = encoded[:, :-1, :] * gates + embedded[:, :, 1:] * (1 - gates)
        else:
            encoded = encoded + embedded[:, :, 1:]
        if self.pool == 'max':
            encoded, _ = (encoded * mask).max(dim=2)
        else:
            encoded = (encoded * mask).sum(dim=2)
        if self.normalize == 'mean':
            lens = mask.sum(2)
            encoded = encoded/(lens+1e-12)
        if self.normalize == 'norm':
            encoded = torch.nn.functional.normalize(encoded, dim=1)
        if proj:
            encoded = proj(encoded)
        if dropout > 0:
            return torch.nn.functional.dropout(encoded, p=dropout, training=self.training).unsqueeze(1)
        else:
            return encoded.unsqueeze(1)

    def encode_subj(self, subj):
        return self._encode(subj, self.entity_embedding, self.entity_encoder_in, None, self.entity_dropout)

    def encode_obj(self, obj):
        return self._encode(obj, self.entity_embedding, self.entity_encoder_in, None, self.entity_dropout)

    def encode_rel(self, rel):
        return self._encode(rel, self.relation_embedding, self.relation_encoder_in, None, self.relation_dropout)

    def get_slot_size(self):
        return self.slot_size


class LSTMRelationEmbedder(TokenBasedRelationEmbedder):

    def __init__(self,
                 entity_slot_size,
                 relation_slot_size,
                 train_data: EntityRelationDatasetMeta,
                 dropout=0.0,
                 entity_dropout=None,
                 relation_dropout=None,
                 encoder_activiation=None,
                 sparse=False,
                 init_std=0.1,
                 normalize='',
                 project_relation=False,
                 ):
        super().__init__(
            entity_slot_size=entity_slot_size,
            relation_slot_size=relation_slot_size,
            train_data=train_data,
            sparse=sparse,
            init_std=init_std,
            normalize=normalize,
        )
        if relation_slot_size is None or relation_slot_size <= 0:
            relation_slot_size = entity_slot_size
        self.relation_slot_size = relation_slot_size
        self.relation_projection = None
        if project_relation:
            self.relation_slot_size = entity_slot_size ** 2
            relation_projection = torch.nn.Linear(relation_slot_size, entity_slot_size**2, bias=False)
            init_core_tensor_std = 1 / (entity_slot_size ** 2 * relation_slot_size * init_std ** 3)
            torch.nn.init.normal_(relation_projection.weight.data, init_core_tensor_std)
            self.relation_projection = Sequential(
                relation_projection,
                torch.nn.BatchNorm1d(entity_slot_size**2)
            )

        self.encoder_activiation = None
        if encoder_activiation is not None and hasattr(torch.nn, encoder_activiation):
            self.encoder_activiation = getattr(torch.nn, encoder_activiation)
        self.entity_encoder_in = torch.nn.LSTM(input_size=entity_slot_size, hidden_size=entity_slot_size, batch_first=True, dropout=dropout)
        self.relation_encoder_in = torch.nn.LSTM(input_size=relation_slot_size, hidden_size=relation_slot_size, batch_first=True, dropout=dropout)
        self.flatten_params_called = False
        self.entity_dropout = entity_dropout if entity_dropout else dropout
        self.relation_dropout = relation_dropout if relation_dropout else dropout

    def _map_to_tokens(self, input, mapper):
        return torch.nn.functional.embedding(
            input.long(), mapper, 0, None,
            0., False, True).view(input.size(0), -1)

    def _encode(self, input, embedder, token_ids, encoder, dropout, norm_func, proj=None):
        if not self.flatten_params_called:
            self.flatten_params_called = True
            self.entity_encoder_in.flatten_parameters()
            self.relation_encoder_in.flatten_parameters()
        input = self._map_to_tokens(input, token_ids)
        last_state = ((input > 0).long().sum(1) - 1)
        return self._encode_tokens(input, embedder, encoder, last_state, norm_func, proj, dropout)

    def _encode_tokens(self, input, embedder, encoder, last_state, norm_func, proj, dropout):
        embedded = embedder(input.long())
        output, hn = encoder(embedded)
        if self.encoder_activiation is not None:
            encoded = self.encoder_activiation(output[range(0,input.size(0)),last_state,:])
        else:
            encoded = output[range(0,input.size(0)),last_state,:]
        if self.normalize == 'batchnorm':
            encoded = norm_func(encoded)
        if proj is not None:
            encoded = proj(encoded)
        if dropout > 0:
            return torch.nn.functional.dropout(encoded, p=dropout, training=self.training).unsqueeze(1)
        else:
            return encoded

    def encode_subj(self, subj):
        return self._encode(subj, self.entity_embedding, self.entity_token_ids, self.entity_encoder_in, self.entity_dropout, self.entity_batchnorm)

    def encode_obj(self, obj):
        return self._encode(obj, self.entity_embedding, self.entity_token_ids, self.entity_encoder_in, self.entity_dropout, self.entity_batchnorm)

    def encode_rel(self, rel):
        return self._encode(rel, self.relation_embedding, self.relation_token_ids, self.relation_encoder_in, self.relation_dropout, self.relation_batchnorm, self.relation_projection)

    def get_slot_size(self):
        return self.slot_size


class LookupTucker3RelationModel(RescalRelationScorer, LookupBaseRelationEmbedder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, project_relation=True)

class LookupComplexRelationModel(ComplexRelationScorer, LookupSimpleRelationEmbedder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class LookupDistmultRelationModel(DistmultRelationScorer, LookupSimpleRelationEmbedder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class UnigramPoolingComplexRelationModel(ComplexRelationScorer, UnigramPoolingRelationEmbedder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BigramPoolingComplexRelationModel(ComplexRelationScorer, BigramPoolingRelationEmbedder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class LSTMComplexRelationModel(ComplexRelationScorer, LSTMRelationEmbedder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class LSTMDistmultRelationModel(DistmultRelationScorer, LSTMRelationEmbedder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DataBiasOnlyEntityModel(DataBiasOnlyEntityScorer, LSTMRelationEmbedder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class DataBiasOnlyRelationModel(DataBiasOnlyRelationScorer, LSTMRelationEmbedder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class LSTMTucker3RelationModel(RescalRelationScorer, LSTMRelationEmbedder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, project_relation=True)


class Models:

    LookupTucker3RelationModel = LookupTucker3RelationModel
    LookupDistmultRelationModel= LookupDistmultRelationModel
    LookupComplexRelationModel= LookupComplexRelationModel

    UnigramPoolingComplexRelationModel= UnigramPoolingComplexRelationModel
    BigramPoolingComplexRelationModel= BigramPoolingComplexRelationModel

    LSTMDistmultRelationModel= LSTMDistmultRelationModel
    LSTMComplexRelationModel= LSTMComplexRelationModel
    LSTMTucker3RelationModel= LSTMTucker3RelationModel

    DataBiasOnlyEntityModel = DataBiasOnlyEntityModel
    DataBiasOnlyRelationModel = DataBiasOnlyRelationModel
