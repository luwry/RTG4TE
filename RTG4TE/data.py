from torch.utils.data import Dataset
from constants import *
from collections import namedtuple
from util import token2sub_tokens
from sentence_transformers import SentenceTransformer, util
import json
import torch
from transformers import BartModel
import random
import re
from random import sample
import numpy as np
import os
instance_fields = [
    'doc_id', 'input_ids', 'attention_mask','decoder_input_chunks', 'input_tokens','document'
]

batch_fields = [
    'doc_ids', 'input_ids', 'attention_masks','decoder_input_chunks', 'input_tokens','document'
]

Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))

class IEDataset(Dataset):
    def __init__(self, config, path, tokenizer, max_length=128, gpu=False):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        :param ignore_title (bool): Ignore sentences that are titles (default=False).
        """
        self.config = config
        self.path = path
        self.retrieval_data = []
        self.data = []
        self.max_length = max_length
        self.gpu = gpu
        # self.sim_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.sim_model = SentenceTransformer("./sim_model")
        self.load_data(tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    
    def load_data(self, tokenizer):
        """Load data from file."""
        overlength_num = title_num = 0
        with open(self.config.retrieval_file, 'r', encoding='utf-8') as r:
            self.retrieval_data = json.loads(r.read())


        input_document_emb_train = []
        decoder_input_chunks_tokenized_train = []

        if config.task == "ree":
            for doc_id, content in self.data.items():
                annotation = content['annotation']
                decoder_input_chunks = self.create_decoder_input_chunks(annotation, tokenizer)
                input_document_emb_train.append(
                    torch.from_numpy(self.sim_model.encode(content['document'], show_progress_bar=False)))
                decoder_input_chunks_tokenized_train.append(decoder_input_chunks)
        elif config.task == "tf":

            for doc_id, content in self.retrieval_data.items():
                type_to_event = dict()
                annotation = content['annotation']
                for event in annotation:
                    event_type = event["incident_type"]
                    type_to_event[event_type] = []
                    type_to_event[event_type].append(event)#将事件添加到对应的事件类型
                for event_type in type_to_event.keys():
                    decoder_input_chunks = self.create_decoder_input_chunks(type_to_event[event_type], tokenizer)
                    input_document_emb_train.append(torch.from_numpy(self.sim_model.encode(content['document'], show_progress_bar=False)))
                    decoder_input_chunks_tokenized_train.append(decoder_input_chunks)#对训练集中包含的事件类型进行编码

                if annotation == []:
                    input_document_emb_train.append(torch.from_numpy(self.sim_model.encode(content['document'], show_progress_bar=False)))
                    decoder_input_chunks_tokenized_train.append([])

        with open(self.path, 'r', encoding='utf-8') as r:
            self.data = json.loads(r.read())
        _data = []
        data = []
        all_data = []
        test_sim = {}
        types = {'attack': 'attack', 'kidnapping': 'kidnapping', 'bombing': 'bombing', 'robbery': 'robbery',
                 'arson': 'arson', 'forced work stoppage': 'forced work stoppage'}

        ###
        for doc_id, content in self.data.items():
            if config.task == "ree":
                annotation = content['annotation']

                decoder_input_chunks = self.create_decoder_input_chunks(annotation, tokenizer)
                decoder_input = tokenizer.decode(decoder_input_chunks)
                input_document_emb = torch.from_numpy(
                    self.sim_model.encode(content['document'], show_progress_bar=False))
            elif config.task == "tf":

                annotation = content['annotation']
                type_to_event = dict()
                for per_event_type in types:
                    type_to_event[per_event_type] = []#
                for per_event_type in types:
                    for per_event in annotation:
                        if per_event["incident_type"] == per_event_type:
                            type_to_event[per_event_type].append(per_event)#
                for per_event_type in type_to_event.keys():
                    decoder_input_chunks = self.create_decoder_input_chunks(type_to_event[per_event_type], tokenizer)
                    ###
                    input_document_emb = torch.from_numpy(self.sim_model.encode(content['document'], show_progress_bar=False))
            if len(self.data) == 1300:
                most_sim = util.semantic_search([input_document_emb], input_document_emb_train, top_k=2)[0][1]
            else:
                most_sim = util.semantic_search([input_document_emb], input_document_emb_train, top_k=2)[0][0]
            most_sim_idx = most_sim['corpus_id']
            most_sim_out_template = decoder_input_chunks_tokenized_train[most_sim_idx]
            most_sim_out_template = tokenizer.decode(most_sim_out_template)
            document = ''.join(most_sim_out_template) + content['document']# w/o prompt template
            if config.task == "ree":
                document = "<PerpInd> [None] </PerpInd><PerpOrg> [None] </PerpOrg><Target> [None] </Target><Victim> [None] </Victim><Weapon> [None] </Weapon>" + document
            elif config.task == "tf":
                document = "<SEP> " + per_event_type + " </SEP>" + document
            input_ids = tokenizer([document], max_length=self.max_length, truncation=True)['input_ids'][0]
            pad_num = self.max_length - len(input_ids)
            attn_mask = [1] * len(input_ids) + [0] * pad_num
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_num
            input_tokens = tokenizer.decode(input_ids)
            instance = Instance(
                doc_id=doc_id,
                input_ids=input_ids,
                attention_mask=attn_mask,
                decoder_input_chunks=decoder_input_chunks,
                input_tokens=input_tokens,
                document=document
            )
            if config.task == "ree":
                data.append(instance)
            elif config.task == "tf":
                all_data.append(instance)
                if len(self.data) == 1300:
                    neg = []
                    pos = []
                    for per_type_event in all_data:
                        if per_type_event.decoder_input_chunks == []:
                            neg.append(per_type_event)
                        else:
                            pos.append(per_type_event)
                    np.random.shuffle(neg)
                    _data.append(pos)
                    _data.append(neg[:3])
                else:
                    _data.append(all_data)

                for example in _data:#
                    for x in example:
                        data.append(x)
        self.data = data#


    def create_decoder_input_chunks(self, templates, tokenizer):

        res = []
        if config.task == "ree":
            for template in templates:
                current_template_chunk = []
                entity_tokens = []
                filler = (
                    f" {AND} ".join([m[0] for m in template['PerpInd']]) if "PerpInd" in template.keys() else NO_ROLE,
                    f" {AND} ".join([m[0] for m in template['PerpOrg']]) if "PerpOrg" in template.keys() else NO_ROLE,
                    f" {AND} ".join([m[0] for m in template['Target']]) if "Target" in template.keys() else NO_ROLE,
                    f" {AND} ".join([m[0] for m in template['Victim']]) if "Victim" in template.keys() else NO_ROLE,
                    f" {AND} ".join([m[0] for m in template['Weapon']]) if "Weapon" in template.keys() else NO_ROLE
                )
                entity_tokens.append(
                    "<PerpInd> {} </PerpInd><PerpOrg> {} </PerpOrg><Target> {} </Target><Victim> {} </Victim><Weapon> {} </Weapon>".format(
                        *filler))

                mention_chunk = []
                for entity_token in entity_tokens:
                    mention_chunk += token2sub_tokens(tokenizer, entity_token)
                current_template_chunk.append(mention_chunk)
                res.append(current_template_chunk)
                res = res[0][0]
        elif config.task == "tf":
            for template in templates:
                if "incident_type" in template.keys():

                    entity_tokens = []

                    filler = (
                        f" {AND} ".join([m for a in template['PerpInd'] for m in a]) if "PerpInd" in template.keys() else NO_ROLE,
                        f" {AND} ".join([m for a in template['PerpOrg'] for m in a]) if "PerpOrg" in template.keys() else NO_ROLE,
                        f" {AND} ".join([m for a in template['Target'] for m in a]) if "Target" in template.keys() else NO_ROLE,
                        f" {AND} ".join( [m for a in template['Victim'] for m in a]) if "Victim" in template.keys() else NO_ROLE,
                        f" {AND} ".join([m for a in template['Weapon'] for m in a]) if "Weapon" in template.keys() else NO_ROLE,
                    )

                    entity_tokens.append("<SEP_T>" + "<SEP> " + template['incident_type'] + " </SEP>" +
                                         "<PerpInd> {} </PerpInd><PerpOrg> {} </PerpOrg><Target> {} </Target><Victim> {} </Victim><Weapon> {} </Weapon>".format(*filler))
                    template_chunk = []
                    for entity_token in entity_tokens:
                        template_chunk += token2sub_tokens(tokenizer, entity_token)
                    res.extend(template_chunk)
        return res


    def collate_fn(self, batch):
        batch_input_ids = []
        batch_attention_masks = []
        batch_decoder_input_chunks = []
        batch_input_tokens = []
        batch_document = []

        doc_ids = [inst.doc_id for inst in batch]
        
        for inst in batch:
            batch_input_ids.append(inst.input_ids)
            batch_attention_masks.append(inst.attention_mask)
            batch_decoder_input_chunks.append(inst.decoder_input_chunks)
            batch_input_tokens.append(inst.input_tokens)
            batch_document.append(inst.document)
        
        if self.gpu:
            batch_input_ids = torch.cuda.LongTensor(batch_input_ids)
            batch_attention_masks = torch.cuda.FloatTensor(batch_attention_masks)

        else:
            batch_input_ids = torch.LongTensor(batch_input_ids)
            batch_attention_masks = torch.FloatTensor(batch_attention_masks)
        
        return Batch(
            doc_ids=doc_ids,
            input_ids=batch_input_ids,
            attention_masks=batch_attention_masks,
            decoder_input_chunks=batch_decoder_input_chunks,
            input_tokens=batch_input_tokens,
            document=batch_document
        )