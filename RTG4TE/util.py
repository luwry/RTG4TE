import torch
import random 
from collections import OrderedDict
import json
from tabulate import tabulate
from typing import Dict, List, Tuple
import re

from constants import *




def token2sub_tokens(tokenizer, token):
    """
    Take in a string value and use tokenizer to tokenize it into subtokens.token:'terrorists'sub_token:'terror'sub_token:'ists'
    Return a list of sub tokens.token:'<PerpInd>'sub_token:'<PerpInd>' res:[50265] res:[26213, 1952]
    """
    res = []
    for sub_token in tokenizer.tokenize(token):
        # make sure it's not an empty string
        if len(sub_token) > 0: 
            res.append(tokenizer.convert_tokens_to_ids(sub_token))
    return res

def format_inputs_outputs(flattened_seqs, tokenizer, use_gpu, max_position_embeddings):
    
    max_seq_len = max([len(seq) for seq in flattened_seqs]) 

    # cannot be greater than position embeddings
    max_seq_len = min(max_position_embeddings, max_seq_len)    

    # create padding & mask
    decoder_input_ids = []
    decoder_masks = []
    decoder_labels = []

    
    for flattened_seq in flattened_seqs:
        
        # minus 1 because mask should match the length of input_ids
        mask = [1] * len(flattened_seq) + [0] * (max_seq_len - len(flattened_seq)-1)

        # padding. 
        flattened_seq += [tokenizer.pad_token_id] * (max_seq_len - len(flattened_seq))
        # flattened_seq += [tokenizer.pad_token_id] * (max_seq_len - len(flattened_seq))
        
        # make sure they do not exceeed max_seq_len -1
        mask = mask[:max_seq_len-1]
        flattened_seq = flattened_seq[:max_seq_len]

        input_ids = flattened_seq[:-1]
        labels = flattened_seq[1:]

        # For some reason, it seems huggingface use -100 to denote tokens that we don't want to compute loss on.
        labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

        decoder_input_ids.append(input_ids)
        decoder_labels.append(labels)
        decoder_masks.append(mask)

    
    # form tensor
    if use_gpu:
        decoder_input_ids = torch.cuda.LongTensor(decoder_input_ids)
        decoder_labels = torch.cuda.LongTensor(decoder_labels)
        decoder_masks = torch.cuda.FloatTensor(decoder_masks)


    else:
        decoder_input_ids = torch.LongTensor(decoder_input_ids)
        decoder_labels = torch.LongTensor(decoder_labels)
        decoder_masks = torch.FloatTensor(decoder_masks)
    
    
    res = {
        'decoder_input_ids': decoder_input_ids,
        'decoder_labels': decoder_labels,
        'decoder_masks': decoder_masks,
    }
    return res



def generate_decoder_inputs_outputs(batch, tokenizer, model, use_gpu, max_position_embeddings, permute_slots=False, task=ROLE_FILLER_ENTITY_EXTRACTION):

    decoder_input_chunks = batch.decoder_input_chunks

    flattened_seqs = []

    for decoder_input_chunk in decoder_input_chunks:

        flatten_entities = []
        # shuffle templates

        for sub_token in decoder_input_chunk:
            flatten_entities.append(sub_token)
        if model.bert.config.name_or_path.startswith('./bart-base') or model.bert.config.name_or_path.startswith('sshleifer/distilbart'):
            flattened_seq = [model.bert.config.decoder_start_token_id, tokenizer.bos_token_id] + flatten_entities + [tokenizer.eos_token_id]#bos_token=’<s>‘ bos_token_id = 0 eos_token = '</s>' eos_token_id = 2
        elif model.bert.config.name_or_path.startswith('t5') or model.bert.config.name_or_path.startswith('google/pegasus') :
            # t5 does not have <s> in the decoded string
            flattened_seq = [model.bert.config.decoder_start_token_id] + flatten_entities + [tokenizer.eos_token_id]
        elif model.bert.config.name_or_path.startswith('./bart-large') or model.bert.config.name_or_path.startswith('sshleifer/distilbart'):
            flattened_seq = [model.bert.config.decoder_start_token_id, tokenizer.bos_token_id] + flatten_entities + [tokenizer.eos_token_id]
        elif model.bert.config.decoder._name_or_path.startswith('roberta'):
            flattened_seq = [model.bert.config.decoder_start_token_id] + flatten_entities + [tokenizer.eos_token_id]
        else:
            print("model name ", model.bert.config)
            raise NotImplementedError
        

        flattened_seqs.append(flattened_seq)


    res = format_inputs_outputs(flattened_seqs, tokenizer, use_gpu, max_position_embeddings)

    return res


def construct_outputs_for_ree(preds, input_documents, doc_ids, tokenizer):
    '''

    input_documents: a list of decoded document (str)

    '''
    res = OrderedDict()
    for predicted_id_sequence, input_document, doc_id in zip(preds, input_documents, doc_ids):
        # convert id to tokens
        predicted_sequence = tokenizer.decode(predicted_id_sequence)  # 把数字id映射回字符串
        # for unknown reason GRIT do this processing for docid
        doc_id = str(int(doc_id.split("-")[0][-1]) * 10000 + int(doc_id.split("-")[-1]))

        # transform into doc
        res[doc_id] = event_templates_to_ree(predicted_sequence, input_document)

    return res
def construct_outputs_for_tf(preds, input_documents, doc_ids, tokenizer):
    '''

    input_documents: a list of decoded document (str)
    
    '''
    res = OrderedDict()
    for doc_id in doc_ids:
        doc_id = str(int(doc_id.split("-")[0][-1]) * 10000 + int(doc_id.split("-")[-1]))
        res[doc_id] = []
    for predicted_id_sequence, input_document, doc_id in zip(preds, input_documents, doc_ids):

        # convert id to tokens
        predicted_sequence = tokenizer.decode(predicted_id_sequence)#把数字id映射回字符串
        # for unknown reason GRIT do this processing for docid
        doc_id = str(int(doc_id.split("-")[0][-1])*10000 + int(doc_id.split("-")[-1]))
        # print(doc_id, predicted_sequence)
        event_template = event_templates_to_tf(predicted_sequence, input_document)

        # transform into doc
        res[doc_id].extend(event_template)
        # print(res[doc_id])

    return res
    

def event_templates_to_tf(event_template_sequences: str, input_document: str):
    '''
    Turns a sequence of event templates into a dictionary
    e.g.
    </s><s><PerpInd>guerrillas</PerpInd><PerpOrg>far right</PerpOrg><Target>santo tomas presidential farm</Target></s><pad><pad>-> {
        'PerpInd':[
            [
                ["guerrillas"],
                
            ]
        ],
        'PerpOrg':[
            [
                ['far right'],
            ]
        ]
        'Target':[
            [
                ['santo tomas presidential farm'],
            ]
        ]
    }
    '''

    # remove the first </s>
    event_template_sequences = event_template_sequences[4:]  #
    try:
        first_eos_index = event_template_sequences.index('</s>')  #
        event_template_sequences = event_template_sequences[:first_eos_index]  # '<s>the guatemala army denied today'
    except:
        pass

    template = []
    prev_slot_name = None
    prev_tag = None  # this is for determining whether a mention is in the same entity cluster as the previous mention
    try:

        if "<SEP_T>" in event_template_sequences:
            if event_template_sequences.startswith('<s>'):
                event_template_sequences = event_template_sequences[len('<s>'):]  # 'the guatemala army denied today'

            event_template_sequences = list(filter(None, re.split("<SEP_T>", event_template_sequences)))
            for event_template_sequence in event_template_sequences:
                res = {
                    'incident_type': [],
                    'PerpInd': [],
                    'PerpOrg': [],
                    'Target': [],
                    'Victim': [],
                    'Weapon': []
                }
                while event_template_sequence:
                    if event_template_sequence.startswith('<SEP>'):
                        if "</SEP>" in event_template_sequence:
                            end_of_SEP_index = event_template_sequence.index("</SEP>")
                            res['incident_type'] = event_template_sequence[len('<SEP>'): end_of_SEP_index].strip()
                            event_template_sequence = event_template_sequence[end_of_SEP_index + len("</SEP>"):]
                        else:
                            event_template_sequence = event_template_sequence[len('<SEP>'):]
                    elif event_template_sequence.startswith(PERP_IND):
                        if END_OF_PERP_IND in event_template_sequence:
                            end_of_entity_index = event_template_sequence.index(END_OF_PERP_IND)
                            mention = event_template_sequence[len(PERP_IND): end_of_entity_index].strip()
                            # print(mention)
                            mention_length = len(mention)
                            event_template_sequence = event_template_sequence[end_of_entity_index + len(END_OF_PERP_IND):]

                            while AND in mention:
                                end_of_mention_index = mention.index(AND)
                                mention_token = mention[: end_of_mention_index].strip()
                                mention = mention[end_of_mention_index + 5:].strip()
                                if mention_token in input_document and [mention_token] not in res['PerpInd']:
                                    res['PerpInd'].append([mention_token])

                            if mention in input_document and mention != '[None]' and [mention] not in res['PerpInd']:
                                res['PerpInd'].append([mention])

                        else:
                            event_template_sequence = event_template_sequence[len(PERP_IND):]


                    elif event_template_sequence.startswith(PERP_ORG):
                        if END_OF_PERP_ORG in event_template_sequence:
                            end_of_entity_index = event_template_sequence.index(END_OF_PERP_ORG)
                            mention = event_template_sequence[len(PERP_ORG): end_of_entity_index].strip()

                            event_template_sequence = event_template_sequence[
                                                      end_of_entity_index + len(END_OF_PERP_ORG):]
                            while AND in mention:
                                end_of_mention_index = mention.index(AND)
                                mention_token = mention[: end_of_mention_index].strip()
                                mention = mention[end_of_mention_index + 5:].strip()
                                if mention_token in input_document and [mention_token] not in res['PerpOrg']:
                                    res['PerpOrg'].append([mention_token])

                            if mention in input_document and mention != '[None]' and [mention] not in res['PerpOrg']:
                                res['PerpOrg'].append([mention])
                                continue

                        else:
                            event_template_sequence = event_template_sequence[len(PERP_ORG):]

                    elif event_template_sequence.startswith(TARGET):
                        if END_OF_TARGET in event_template_sequence:
                            end_of_entity_index = event_template_sequence.index(END_OF_TARGET)
                            mention = event_template_sequence[len(TARGET): end_of_entity_index].strip()
                            mention_length = len(mention)
                            event_template_sequence = event_template_sequence[end_of_entity_index + len(END_OF_TARGET):]
                            while AND in mention:
                                end_of_mention_index = mention.index(AND)
                                mention_token = mention[: end_of_mention_index].strip()
                                mention = mention[end_of_mention_index + 5:].strip()
                                if mention_token in input_document and [mention_token] not in res['Target']:
                                    res['Target'].append([mention_token])
                            if mention in input_document and mention != '[None]' and [mention] not in res['Target']:
                                res['Target'].append([mention])

                        else:
                            event_template_sequence = event_template_sequence[len(TARGET):]

                    elif event_template_sequence.startswith(VICTIM):
                        if END_OF_VICTIM in event_template_sequence:
                            end_of_entity_index = event_template_sequence.index(END_OF_VICTIM)
                            mention = event_template_sequence[len(VICTIM): end_of_entity_index].strip()
                            mention_length = len(mention)
                            event_template_sequence = event_template_sequence[end_of_entity_index + len(END_OF_VICTIM):]
                            while AND in mention:
                                end_of_mention_index = mention.index(AND)
                                mention_token = mention[: end_of_mention_index].strip()
                                mention = mention[end_of_mention_index + 5:].strip()
                                if mention_token in input_document and [mention_token] not in res['Victim']:
                                    res['Victim'].append([mention_token])
                            if mention in input_document and mention != '[None]' and [mention] not in res['Victim']:
                                res['Victim'].append([mention])

                        else:
                            event_template_sequence = event_template_sequence[len(VICTIM):]

                    elif event_template_sequence.startswith(WEAPON):
                        if END_OF_WEAPON in event_template_sequence:
                            end_of_entity_index = event_template_sequence.index(END_OF_WEAPON)
                            mention = event_template_sequence[len(WEAPON): end_of_entity_index].strip()
                            # mention_length = len(mention)
                            event_template_sequence = event_template_sequence[end_of_entity_index + len(END_OF_WEAPON):]
                            while AND in mention:
                                end_of_mention_index = mention.index(AND)
                                mention_token = mention[: end_of_mention_index].strip()
                                mention = mention[end_of_mention_index + 5:].strip()
                                if mention_token in input_document and [mention_token] not in res['Weapon']:
                                    res['Weapon'].append([mention_token])
                            if mention in input_document and mention != '[None]' and [mention] not in res['Weapon']:
                                res['Weapon'].append([mention])

                        else:
                            event_template_sequence = event_template_sequence[len(WEAPON):]


                    else:
                        # if nothing match, reduce the sequence length by 1 and move forward
                        event_template_sequence = event_template_sequence[1:]
                if res not in template:
                    template.append(res)



    except Exception:

        print(event_template_sequences)
    return template


def event_templates_to_ree(event_template_sequence: str, input_document: str):
    '''
    Turns a sequence of event templates into a dictionary
    e.g.
    </s><s><PerpInd>guerrillas</PerpInd><PerpOrg>far right</PerpOrg><Target>santo tomas presidential farm</Target></s><pad><pad>-> {
        'PerpInd':[
            [
                ["guerrillas"],

            ]
        ],
        'PerpOrg':[
            [
                ['far right'],
            ]
        ]
        'Target':[
            [
                ['santo tomas presidential farm'],
            ]
        ]
    }
    '''

    # remove the first </s>
    event_template_sequence = event_template_sequence[4:]  #
    try:
        first_eos_index = event_template_sequence.index('</s>')  #
        event_template_sequence = event_template_sequence[:first_eos_index]  # '<s>the guatemala army denied today'
    except:
        pass
    res = {
        'PerpInd': [],
        'PerpOrg': [],
        'Target': [],
        'Victim': [],
        'Weapon': []
    }
    prev_slot_name = None
    prev_tag = None  # this is for determining whether a mention is in the same entity cluster as the previous mention
    try:
        while event_template_sequence:
            if event_template_sequence.startswith('<s>'):
                event_template_sequence = event_template_sequence[len('<s>'):]  # 'the guatemala army denied today'
                continue

            elif event_template_sequence.startswith(PERP_IND):
                if END_OF_PERP_IND in event_template_sequence:
                    end_of_entity_index = event_template_sequence.index(END_OF_PERP_IND)
                    mention = event_template_sequence[len(PERP_IND): end_of_entity_index].strip()
                    # print(mention)
                    mention_length = len(mention)
                    event_template_sequence = event_template_sequence[end_of_entity_index + len(END_OF_PERP_IND):]
                    while AND in mention:
                        end_of_mention_index = mention.index(AND)
                        mention_token = mention[: end_of_mention_index].strip()
                        mention = mention[end_of_mention_index + 5:].strip()
                        if mention_token in input_document:
                            res['PerpInd'].append([mention_token])

                    if mention in input_document and mention != '[None]':
                        res['PerpInd'].append([mention])

                else:
                    event_template_sequence = event_template_sequence[len(PERP_IND):]


            elif event_template_sequence.startswith(PERP_ORG):
                if END_OF_PERP_ORG in event_template_sequence:
                    end_of_entity_index = event_template_sequence.index(END_OF_PERP_ORG)
                    mention = event_template_sequence[len(PERP_ORG): end_of_entity_index].strip()

                    event_template_sequence = event_template_sequence[end_of_entity_index + len(END_OF_PERP_ORG):]
                    while AND in mention:
                        end_of_mention_index = mention.index(AND)
                        mention_token = mention[: end_of_mention_index].strip()
                        mention = mention[end_of_mention_index + 5:].strip()
                        if mention_token in input_document:
                            res['PerpOrg'].append([mention_token])

                    if mention in input_document and mention != '[None]':
                        res['PerpOrg'].append([mention])
                        continue

                else:
                    event_template_sequence = event_template_sequence[len(PERP_ORG):]

            elif event_template_sequence.startswith(TARGET):
                if END_OF_TARGET in event_template_sequence:
                    end_of_entity_index = event_template_sequence.index(END_OF_TARGET)
                    mention = event_template_sequence[len(TARGET): end_of_entity_index].strip()
                    mention_length = len(mention)
                    event_template_sequence = event_template_sequence[end_of_entity_index + len(END_OF_TARGET):]
                    while AND in mention:
                        end_of_mention_index = mention.index(AND)
                        mention_token = mention[: end_of_mention_index].strip()
                        mention = mention[end_of_mention_index + 5:].strip()
                        if mention_token in input_document:
                            res['Target'].append([mention_token])
                    if mention in input_document and mention != '[None]':
                        res['Target'].append([mention])
                        continue

                else:
                    event_template_sequence = event_template_sequence[len(TARGET):]


            elif event_template_sequence.startswith(VICTIM):
                if END_OF_VICTIM in event_template_sequence:
                    end_of_entity_index = event_template_sequence.index(END_OF_VICTIM)
                    mention = event_template_sequence[len(VICTIM): end_of_entity_index].strip()
                    mention_length = len(mention)
                    event_template_sequence = event_template_sequence[end_of_entity_index + len(END_OF_VICTIM):]
                    while AND in mention:
                        end_of_mention_index = mention.index(AND)
                        mention_token = mention[: end_of_mention_index].strip()
                        mention = mention[end_of_mention_index + 5:].strip()
                        if mention_token in input_document:
                            res['Victim'].append([mention_token])
                    if mention in input_document and mention != '[None]':
                        res['Victim'].append([mention])
                        continue
                else:
                    event_template_sequence = event_template_sequence[len(VICTIM):]

            elif event_template_sequence.startswith(WEAPON):
                if END_OF_WEAPON in event_template_sequence:
                    end_of_entity_index = event_template_sequence.index(END_OF_WEAPON)
                    mention = event_template_sequence[len(WEAPON): end_of_entity_index].strip()
                    # mention_length = len(mention)
                    event_template_sequence = event_template_sequence[end_of_entity_index + len(END_OF_WEAPON):]
                    while AND in mention:
                        end_of_mention_index = mention.index(AND)
                        mention_token = mention[: end_of_mention_index].strip()
                        mention = mention[end_of_mention_index + 5:].strip()
                        if mention_token in input_document:
                            res['Weapon'].append([mention_token])
                    if mention in input_document and mention != '[None]':
                        res['Weapon'].append([mention])

                else:
                    event_template_sequence = event_template_sequence[len(WEAPON):]


            else:
                # if nothing match, reduce the sequence length by 1 and move forward
                event_template_sequence = event_template_sequence[1:]

    except Exception as e:

        print(event_template_sequence)
    return res


def read_tf_gold_file(file: str):
    golds = OrderedDict()
    with open(file, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            docid = str(int(line["docid"].split("-")[0][-1]) * 10000 + int(line["docid"].split("-")[-1]))

            extracts_raw = line["templates"]
            extracts = []
            for template in extracts_raw:
                template_raw = OrderedDict()
                for role, entitys_raw in template.items():
                    template_raw[role] = []
                    if role == "incident_type":
                        template_raw[role] = entitys_raw
                    else:
                        for entity_raw in entitys_raw:
                            entity = []
                            for mention_offset_pair in entity_raw:
                                entity.append(mention_offset_pair[0])
                            if entity:
                                template_raw[role].append(entity)
                extracts.append(template_raw)
            golds[docid] = extracts
    return golds
def read_grit_gold_file(file: str):
    golds = OrderedDict()
    with open(file, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                docid = str(int(line["docid"].split("-")[0][-1])*10000 + int(line["docid"].split("-")[-1]))

                extracts_raw = line["extracts"]

                extracts = OrderedDict()
                for role, entitys_raw in extracts_raw.items():
                    extracts[role] = []
                    for entity_raw in entitys_raw:
                        entity = []
                        for mention_offset_pair in entity_raw:
                            entity.append(mention_offset_pair[0])
                        if entity:
                            extracts[role].append(entity)
                golds[docid] = extracts
    return golds



def construct_table(result):
    def format_string(score):
        return f'{score*100:.2f}'

    table = [["role", "prec", "rec",'f1']]
    for key, values in result.items():
        table.append( [key, format_string(values['p']), format_string(values['r']), format_string(values['f1']) ])
    
    return tabulate(table, headers="firstrow", tablefmt="grid")

def get_best_score(log_file: str, role: str):

     with open(log_file, 'r', encoding='utf-8') as r:
        config = r.readline()

        best_scores = []
        best_dev_score = 0
        for line in r:
            record = json.loads(line)
            dev = record['dev']
            test = record['test']
            epoch = record['epoch']
            
            if dev[role]['f1'] > best_dev_score:
                best_dev_score = dev[role]['f1']
                best_scores = [dev, test, epoch]

        print('Best Epoch: {}'.format(best_scores[-1]))
        
        best_dev, best_test, epoch = best_scores
        print("Dev")
        print(construct_table(best_dev))
        print("Test")
        print(construct_table(best_test))
