# ------------------------------------------------------------------------
# HOTR official code : hotr/models/detr.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR & HOTR model and criterion classes.
"""
import os
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from torch import nn
from termcolor import colored

from hotr.util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from .hotr import HOTR
from .backbone import build_backbone
from .detr_matcher import build_matcher
from .hotr_matcher import build_hoi_matcher
from .transformer import build_transformer, build_hoi_transformer
from .criterion import SetCriterion
from .post_process import PostProcess
from .feed_forward import MLP
from utils.log_helper import default_logger as logger
from .known_novel_split import known_objects, novel_objects
from  hotr.data.datasets.builtin_meta import COCO_CATEGORIES

def load_semantic_embeddings(semantic_corpus, classes, precomputed_semantic_embs=None):
    """
    Load precomputed semantic embeddings if it exists. Otherwise, extract it from corpus.
    Args:
        semantic_corpus (str)
        classes (List[str])
        precomputed_semantic_embs (str)
    Returns:
        class_embs_dict (Dict[str: np.array])
    """
    # Prepare the semantic embeddings
    to_compute_semantic_embs = True
    if os.path.isfile(precomputed_semantic_embs):
        with open(precomputed_semantic_embs, "rb") as f:
            precomputed_embs_dict = pickle.load(f)
        # Check if novel classes exist in precomputed embs
        if all(x in precomputed_embs_dict.keys() for x in classes):
            return precomputed_embs_dict

    if to_compute_semantic_embs:
        # We take the average for classes e.g. "hot dog", "parking meter".
        word_embs_dict = {x: None for cls in classes for x in cls.split(" ")}
        with open(semantic_corpus, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.split("\n")[0].split(" ")
                word = line[0]
                if word in word_embs_dict:
                    emb = np.asarray([float(x) for x in line[1:]])
                    word_embs_dict[word] = emb
                if all([v is not None for k, v in word_embs_dict.items()]):
                    # Break if all words have found its embedding.
                    break
        # check all words have a corresponding semantic embeddings
        none_embs = [x for x, emb in word_embs_dict.items() if emb is None]
        if len(none_embs) > 0:
            msg = "Some classes (words) are not in the corpus and will be skipped in inference:\n"
            msg += "\n".join("  " + colored(x, "blue") for x in none_embs)
            logger.info(msg)
        # Remove none classes
        def is_valid(cls, none_embs):
            for x in cls.split(" "):
                if x in none_embs:
                    return False
            return True
        classes = [x for x in classes if is_valid(x, none_embs)]
        
        class_embs_dict = {}
        for cls in classes:
            emb = [word_embs_dict[x] for x in cls.split(" ") if word_embs_dict[x] is not None]
            emb = np.stack(emb, axis=0).mean(axis=0)
            class_embs_dict[cls] = emb

    # Save semantic embeddings to avoid repeated computations.
    if os.path.isfile(precomputed_semantic_embs):
        with open(precomputed_semantic_embs, "rb") as f:
            precomputed_embs_dict = pickle.load(f)
        precomputed_embs_dict.update(class_embs_dict)
        with open(precomputed_semantic_embs, "wb") as f:
            pickle.dump(precomputed_embs_dict, f)
    else:
        with open("./data/precomputed_semantic_embeddings.pkl", "wb") as f:
            pickle.dump(class_embs_dict, f)
    return class_embs_dict

class ZeroShotWordEmbedding(nn.Module):
    """
    Zero-shot predictors for discovering objects from novel categories.
    """
    def __init__(self, known_classes, novel_classes):
        super(ZeroShotWordEmbedding, self).__init__()
        SEMANTIC_CORPUS = "data/Glove/glove.6B.300d.txt"
        PRECOMPUTED_SEMANTIC_EMBEDDINGS = "data/object_word_semantic_embeddings/precomputed_semantic_embeddings.pkl"
        self.precomputed_semantic_embs = PRECOMPUTED_SEMANTIC_EMBEDDINGS
        self.semantic_corpus           = SEMANTIC_CORPUS
        self._init_embs(known_classes, novel_classes)
    
    def _init_embs(self, known_classes, novel_classes):
        """
        Initilize semantic embeddings for classes.
        """
        # laading semantic word embeddings.
        class_embs_dict = load_semantic_embeddings(
            self.semantic_corpus,
            known_classes + novel_classes,
            self.precomputed_semantic_embs,
        )
        assert all([x in class_embs_dict for x in known_classes])
        self.known_classes = known_classes
        self.novel_classes = [x for x in novel_classes if x in class_embs_dict]

        self.known_class_embs = torch.stack([
            torch.as_tensor(class_embs_dict[x],dtype=torch.float32) for x in known_classes
        ], dim=0)

        if len(self.novel_classes) == 0:
            return
        self.novel_class_embs = torch.stack([
            torch.as_tensor(class_embs_dict[x],dtype=torch.float32) for x in novel_classes if x in class_embs_dict
        ], dim=0)

    def get_embs(self):
        """ Return known and novel class embeddings. """
        return self.known_class_embs, self.novel_class_embs

        
class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, valid_obj_ids, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.valid_obj_ids = valid_obj_ids
        # TODO
        zero_shot_embedder = ZeroShotWordEmbedding(known_objects,novel_objects)
        self.known_class_embs, self.novel_class_embs = zero_shot_embedder.get_embs()
        assert self.known_class_embs.shape[1] == self.novel_class_embs.shape[1], "Embedding dimensions do not match"
        self.class_embed_dim = self.known_class_embs.shape[1]
        self.known_ids = []
        self.unknown_ids=[]
        # self.class_embed = torch.ones(num_classes + 1, self.class_embed_dim)
        self.class_embed = torch.randn(num_classes + 1, self.class_embed_dim)
        
        for dict_item in COCO_CATEGORIES:
            if dict_item.get('name') in known_objects:
                self.known_ids.append(dict_item.get('id'))
            elif dict_item.get('name') in novel_objects:
                self.unknown_ids.append(dict_item.get('id'))
        self.class_embed[self.known_ids,:] = self.known_class_embs
        self.class_embed[self.unknown_ids,:] = self.novel_class_embs
        self.class_embed = self.class_embed.cuda()
        # TODO
        #  self.background_embed = torch.nn.Parameter(torch.zeros(1, self.class_embed_dim)).cuda()
        # self.non_object_embed = torch.nn.Parameter(torch.zeros(1, self.class_embed_dim)).cuda()
        # self.class_embed = torch.cat([self.background_embed, self.known_class_embs, self.novel_class_embs,self.non_object_embed], dim=0) # torch.Size([80,300])
        # self.class_embed = torch.nn.Parameter(torch.FloatTensor(num_classes + 1, self.class_embed_dim)).cuda()
        # self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def build(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        valid_obj_ids = args.valid_obj_ids,   
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    
    losses = ['labels', 'boxes', 'cardinality'] if args.frozen_weights is None else []

    if args.HOIDet:
        hoi_matcher = build_hoi_matcher(args)
        hoi_losses = []
        hoi_losses.append('pair_labels')
        hoi_losses.append('pair_actions')
        if args.dataset_file == 'hico-det': 
            hoi_losses.append('pair_targets')
            hoi_losses.append('pair_interactions')
        hoi_weight_dict={}
        hoi_weight_dict['loss_hidx'] = args.hoi_idx_loss_coef
        hoi_weight_dict['loss_oidx'] = args.hoi_idx_loss_coef
        hoi_weight_dict['loss_act'] = args.hoi_act_loss_coef
        hoi_weight_dict['loss_inter'] = args.hoi_interaction_loss_coef
        
        if args.dataset_file == 'hico-det': hoi_weight_dict['loss_tgt'] = args.hoi_tgt_loss_coef
        if args.hoi_aux_loss:
            hoi_aux_weight_dict = {}
            for i in range(args.hoi_dec_layers):
                hoi_aux_weight_dict.update({k + f'_{i}': v for k, v in hoi_weight_dict.items()})
            hoi_weight_dict.update(hoi_aux_weight_dict)

        criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=hoi_weight_dict,
                                 eos_coef=args.eos_coef, losses=losses, num_actions=args.num_actions,
                                 HOI_losses=hoi_losses, HOI_matcher=hoi_matcher, args=args)

        interaction_transformer = build_hoi_transformer(args) # if (args.share_enc and args.pretrained_dec) else None

        kwargs = {}
        if args.dataset_file == 'hico-det': kwargs['return_obj_class'] = args.valid_obj_ids
        
        model = HOTR(
            detr=model,
            num_hoi_queries=args.num_hoi_queries,
            num_actions=args.num_actions,
            interaction_transformer=interaction_transformer,
            freeze_detr=(args.frozen_weights is not None),
            share_enc=args.share_enc,
            pretrained_dec=args.pretrained_dec,
            temperature=args.temperature,
            hoi_aux_loss=args.hoi_aux_loss,
            **kwargs # only return verb class for HICO-DET dataset
        )
        postprocessors = {'hoi': PostProcess(args.HOIDet)}
    else:
        criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses)
        postprocessors = {'bbox': PostProcess(args.HOIDet)}
    criterion.to(device)

    return model, criterion, postprocessors