import argparse
import json
import nltk
# these are for splitting doctext to sentences 
nltk.download('punkt')
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def process_entities(entities):

    '''
    [
        [
            ['terrorists', 102]
        ]
    ]
    ->[['terrorists']]

    [
        [
            ['farabundo marti national liberation front', 120],
            ['fmln', 163]
        ]
    ]
    ->[['farabundo marti national liberation front', 'fmln']]

    '''

    res = []
    for entity in entities:

        # take only the string 
        res.append([mention[0] for mention in entity])     

    return res

def convert(doc, capitalize=False):
    '''
    doc: a dictionary that has the following format:
    {'docid': 'DEV-MUC3-0001',
    'doctext': "the arce battalion command has reported that about 50 peasants of various ages have been kidnapped by terrorists of the farabundo marti national liberation front (fmln) in san miguel department.  according to that garrison, the mass kidnapping took place on 30 december in san luis de la reina.  the source added that the terrorists forced the individuals, who were taken to an unknown location, out of their residences, presumably to incorporate them against their will into clandestine groups.    meanwhile, three subversives were killed and seven others were wounded during clashes yesterday in usulutan and morazan departments.  the atonal battalion reported that one extremist was killed and five others were wounded during a clash yesterday afternoon near la esperanza farm, santa elena jurisdiction, usulutan department.    it was also reported that a soldier was wounded and taken to the military hospital in this capital.    the same military unit reported that there was another clash that resulted in one dead terrorist and the seizure of various kinds of war materiel near san rafael farm in the same town.    in the country's eastern region, military detachment no.4 reported that a terrorist was killed and two others were wounded during a clash in la ranera stream, san carlos, morazan department.  an m-16 rifle, cartridge clips, and ammunition were seized there.    meanwhile, the 3d infantry brigade reported that ponce battalion units found the decomposed body of a subversive in la finca hill, san miguel.  an m-16 rifle, five grenades, and material for the production of explosives were found in the same place.  the brigade, which is headquartered in san miguel, added that the seizure was made yesterday morning.     national guard units guarding the las canas bridge, which is on the northern trunk highway in apopa, this morning repelled a terrorist attack that resulted in no casualties.  the armed clash involved mortar and rifle fire and lasted 30 minutes.  members of that security group are combing the area to determine the final outcome of the fighting.",
    'templates': [{'incident_type': 'kidnapping',
    'PerpInd': [[['terrorists', 102]]],
    'PerpOrg': [[['farabundo marti national liberation front', 120], ['fmln', 163]]],
    'Target': [], 'Victim': [], 'Weapon': []},
    {'incident_type': 'attack',
    'PerpInd': [[['terrorist', 102]]],
    'PerpOrg': [],
    'Target': [[['las canas bridge', 1774]]],
    'Victim': [],
    'Weapon': [[['rifle', 1322]], [['mortar', 1940]]]}]}

    capitalize: whether to capitalize doctext or not
    '''

    res = {
        'docid': doc['docid'], 
        'document': doc['doctext'], # the raw text document.
        'annotation': [] # A list of templates. In role-filler entity extraction, we only have one template for each don't care about this.       
    }

    if capitalize:
        # split doctext into sentences
        sentences = sent_tokenizer.tokenize(doc['doctext'])
        capitalized_doctext = ' '.join([sent.capitalize() for sent in sentences])
        res['document'] = capitalized_doctext



    # TODO: add "tags" in the document
    # res['document'] = doc_text_no_n

    annotation = doc['templates']
    for template in annotation:
        template_dic = {}
        for role, entities in template.items():
            # make sure entities is not an empty list
            if entities:
                # make sure res['annotation'] has one dictionary
                if role == "incident_type":
                    template_dic[role] = entities
                else:
                    template_dic[role] = process_entities(entities)
        if template_dic['incident_type'] in ['kidnapping', 'attack', 'bombing', "arson", 'robbery']:
            res['annotation'].append(template_dic)
    return res

if __name__ == '__main__':
    
    p = argparse.ArgumentParser("Convert GRIT input data into ours format.")
    
    p.add_argument('--input_path', default="./data/train.json", type=str, help="input file in GRIT format.")
    p.add_argument('--output_path', default="./data/tf_train.json", type=str, help="path to store the output json file.")
    p.add_argument('--capitalize',action="store_true", help="whether to capitalize the first char of each sentence")
    args = p.parse_args()

    with open(args.input_path, 'r') as f:
        grit_inputs = [json.loads(l) for l in f.readlines()]

    all_processed_doc = dict()

    # iterate thru and process all grit documents 
    for grit_doc in grit_inputs:
        
        processed = convert(grit_doc, args.capitalize)
        doc_id = processed.pop('docid')
        all_processed_doc[doc_id] = processed
    
    with open(args.output_path, 'w') as f:
        f.write(json.dumps(all_processed_doc))
