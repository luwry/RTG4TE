# -*-coding:utf-8-*-
from constants import *


decoder_input='</s><s><SEP_T><SEP> attack </SEP><PerpInd> one of the residence guards </PerpInd><PerpOrg> [None] </PerpOrg>' \
              '<Target> [None] </Target><Victim> manuel antonio rugama </Victim><Weapon> [None] </Weapon><SEP_T><SEP> attack </SEP>' \
              '<PerpInd> jose jesus pena [and] jose jesus pena--alleged chief of security for the nicaraguan embassy in tegucigalpa </PerpInd>' \
              '<PerpOrg> nicaraguan embassy </PerpOrg><Target> [None] </Target><Victim> manuel antonio rugama </Victim><Weapon> [None] </Weapon>'

PERP_IND='<PerpInd>'
END_OF_PERP_IND='</PerpInd>'
PERP_ORG='<PerpOrg>'
END_OF_PERP_ORG='</PerpOrg>'
TARGET='<Target>'
END_OF_TARGET='</Target>'
VICTIM='<Victim>'
END_OF_VICTIM='</Victim>'
WEAPON='<Weapon>'
END_OF_WEAPON='</Weapon>'
NONE='[None]'
AND='[and]'
REE_ROLES = [PERP_IND, END_OF_PERP_IND, PERP_ORG, END_OF_PERP_ORG, TARGET, END_OF_TARGET, VICTIM, END_OF_VICTIM, WEAPON, END_OF_WEAPON, NONE]

import re
input_document = "</s><s><SEP_T><SEP> attack </SEP><PerpInd> [None] </PerpInd><PerpOrg> [None] </PerpOrg><Target> [None] </Target>" \
                 "<Victim> luis carlos galan sarmiento </Victim><Weapon> [None] </Weapon><SEP_T><SEP> attack </SEP><PerpInd> [None] </PerpInd><PerpOrg> [None] </PerpOrg><Target> [None] </Target><Victim> luis carlos galan sarmiento </Victim><Weapon> [None] </Weapon></s>"
# event_template_sequences = '</s><s><SEP_T><SEP> kidnapping </SEP><PerpInd> terrorists </PerpInd><PerpOrg> farabundo marti national liberation front </PerpOrg><Target> [None] </Target><Victim> [None] </Victim><Weapon> [None] </Weapon><SEP_T><SEP> attack </SEP><PerpInd> terrorist </PerpInd><PerpOrg> [None] </PerpOrg><Target> las canas bridge </Target><Victim> [None] </Victim><Weapon> rifle [and] mortar </Weapon></s>'
event_template_sequences = "</s><s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad> national liberation army [and] eln commando </PerpOrg>" \
                           "<Target> [None] </Target><Victim> miguel antonio avila barreto </Victim><Weapon> [None] </Weapon><SEP_T><SEP> kidnapping </SEP>" \
                           "<PerpInd> national liberation army (eln) guerrillas [and] guerrillas [and] eln commando [and] commando </PerpInd>" \
                           "<PerpOrg> national liberation army [and] eln commando [and] eln commando [and] national liberation army [and] eln </PerpOrg><Target> [None] </Target>" \
                           "<Victim> guillermo rodriguez velasquez [and] miguel antonio avila barreto </Victim><Weapon> [None] </Weapon><SEP_T><SEP> kidnapping </SEP><PerpInd> national liberation army (eln) " \
                           "guerrillas today kidnapped the antioquia colombian civil defense (dcc) commander.    the police said major guillermo rodriguez velasquez, retired, " \
                           "was intercepted by an eln commando while traveling from doradal to medellin with sergeant miguel antonio avila barreto, also retired, who is the department's dcc recruiter.    " \
                           "the two dcc officials were returning to medellin when they were kidnapped 1 km from doradal.</s> "


# event_template_sequences = '</s><s><SEP_T><SEP> kidnapping </SEP><PerpInd> five unidentified individuals [and] unidentified individuals </PerpInd><PerpOrg> [None] </PerpOrg>' \
#                            '<Target> [None] </Target><Victim> luis morales </Victim><Weapon> [None] </Weapon>'
# remove the first </s>
event_template_sequences = event_template_sequences[4:]#
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
                template.append(res)


except Exception:

    print(event_template_sequence)
"""
from collections import OrderedDict
event_template = [{'incident_type': 'kidnapping', 'PerpInd': [['five unidentified individuals'], ['unidentified individuals']], 'PerpOrg': [], 'Target': [], 'Victim': [['luis morales']], 'Weapon': []}]
event_template = [{'incident_type': 'attack', 'PerpInd': [['one of the residence guards']], 'PerpOrg': [], 'Target': [], 'Victim': [['manuel antonio rugama']], 'Weapon': []}, {'incident_type': 'attack', 'PerpInd': [['jose jesus pena'], ['jose jesus pena--alleged chief of security for the nicaraguan embassy in tegucigalpa']], 'PerpOrg': [['nicaraguan embassy']], 'Target': [], 'Victim': [['manuel antonio rugama']], 'Weapon': []}]
doc_ids = ["TST3-MUC4-0001"]
res = OrderedDict()
for doc_id in doc_ids:
    doc_id = str(int(doc_id.split("-")[0][-1]) * 10000 + int(doc_id.split("-")[-1]))
    res[doc_id] = []

res[doc_id].extend(event_template)
"""
