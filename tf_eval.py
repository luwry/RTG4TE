import numpy as np
import re
import string
import json
import argparse
from collections import OrderedDict
import itertools

tag2role = OrderedDict(
    {'incident_type': 'incident_type', 'perp_individual_id': "PerpInd", 'perp_organization_id': "PerpOrg",
     'phys_tgt_id': "Target", 'hum_tgt_name': "Victim", 'incident_instrument_id': "Weapon"})


def normalize_string(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


def matching(c1, c2):
    # similarity: if c2 (pred) is subset of c1 (gold) return 1
    for m in c2:#遍历黄金标注的实体
        if m not in c1:
            return 0
    return 1#当预测的实体存在于黄金标注的实体中则返回1


def is_valid_mapping(mapping):#mapping：{0: 0, 1: 2}
    reverse_mapping = {}
    for k in mapping:
        v = mapping[k]
        if v not in reverse_mapping:
            reverse_mapping[v] = [k]
        else:
            reverse_mapping[v].append(k)#黄金标注的序列序号对应预测序列序号

    for v in reverse_mapping:#v表示预测值
        if v == -1: continue#-1表明不存在事件，则不进行计算
        if len(reverse_mapping[v]) > 1:#黄金标注的事件对应的预测正确的事件只能有一个
            return False

    return True


def score(mapping, pred, gold):
    ex_result = OrderedDict()
    all_keys = list(role for _, role in tag2role.items()) + ["micro_avg", "target"]
    for key in all_keys:
        ex_result[key] = {"p_num": 0, "p_den": 0, "r_num": 0, "r_den": 0, "p": 0, "r": 0, "f1": 0}
    # if invalid mapping, return 0
    # if not is_valid_mapping(mapping):
    #     return ex_result

    mapped_temp_pred = []
    mapped_temp_gold = []
    for pred_temp_idx in mapping:
        gold_temp_idx = mapping[pred_temp_idx]
        if type(pred[pred_temp_idx]["incident_type"]) != str:
            pred[pred_temp_idx]["incident_type"] = "attack"
        if gold_temp_idx != -1 and pred[pred_temp_idx]["incident_type"] in gold[gold_temp_idx][
            "incident_type"]:  # attach vs attach / bombing 先判断事件类型是否预测正确
            mapped_temp_pred.append(pred_temp_idx)
            mapped_temp_gold.append(gold_temp_idx)
            pred_temp, gold_temp = pred[pred_temp_idx], gold[gold_temp_idx]

            # prec对事件角色对应的事件元素进行预测
            for role in pred_temp.keys():
                if role == "incident_type":
                    # if pred_temp["incident_type"] in ['attack']:
                    #     ex_result['target']["p_den"] += 1
                    #     ex_result['target']["p_num"] += 1
                    ex_result[role]["p_den"] += 1
                    ex_result[role]["p_num"] += 1
                    continue
                for entity_pred in pred_temp[role]:#遍历预测的角色下存在的实体，一个列表中只包含一个实体 target：[['presidential farm'], ['farm']]
                    ex_result[role]["p_den"] += 1#预测的实体个数，当预测的实体不为空（eg'PerpOrg'= []）时，预测的实体个数+1
                    correct = False
                    for entity_gold in gold_temp[role]:
                        # import ipdb; ipdb.set_trace()
                        if matching(entity_gold, entity_pred):#预测的实体['guerrillas']与黄金标注的实体['guerrillas', 'guerrilla column']进行匹配
                            correct = True
                    if correct:
                        ex_result[role]["p_num"] += 1#预测的实体中预测正确的个数。预测正确时，预测正确的实体个数+1

            # recall
            for role in gold_temp.keys():
                if role == "incident_type":
                    # if pred_temp["incident_type"] in ['attack']:
                    #     ex_result['target']["r_den"] += 1
                    #     ex_result['target']["r_num"] += 1
                    ex_result[role]["r_den"] += 1#根据循环条件，需要识别的事件类型个数和识别正确的事件类型个数分别+1
                    ex_result[role]["r_num"] += 1
                    continue
                for entity_gold in gold_temp[role]:#排除掉黄金标注为空的元素角色
                    ex_result[role]["r_den"] += 1#遍历黄金标注的实体，需要识别的实体个数+1
                    correct = False
                    for entity_pred in pred_temp[role]:
                        if matching(entity_gold, entity_pred):
                            correct = True
                    if correct:
                        ex_result[role]["r_num"] += 1#同理可得黄金标注的实体中预测正确的实体个数

    # spurious当预测的事件类型与黄金标注的事件类型不一致，统计预测错误的实体个数，统计预测的实体个数
    for pred_temp_idx in range(len(pred)):
        pred_temp = pred[pred_temp_idx]
        if pred_temp_idx not in mapped_temp_pred:#将匹配过的事件不再进行匹配
            for role in pred_temp:
                if role == "incident_type":
                #     if pred_temp["incident_type"] in ['attack']:
                #         ex_result['target']["p_den"] += 1
                    ex_result[role]["p_den"] += 1
                    continue
                for entity_pred in pred_temp[role]:
                    ex_result[role]["p_den"] += 1

    # missing 没有预测出来的黄金标注事件进行统计 统计需要识别的实体个数
    for gold_temp_idx in range(len(gold)):
        gold_temp = gold[gold_temp_idx]
        if gold_temp_idx not in mapped_temp_gold:
            for role in gold_temp:
                if role == "incident_type":
                    # if gold_temp["incident_type"] in ['attack']:
                    #     ex_result['target']["r_den"] += 1
                    ex_result[role]["r_den"] += 1
                    continue
                for entity_gold in gold_temp[role]:
                    ex_result[role]["r_den"] += 1#['santo tomas presidential farm', 'presidential farm']计算为1个实体

    ex_result["micro_avg"]["p_num"] = sum(ex_result[role]["p_num"] for _, role in tag2role.items())#预测的实体中预测正确的实体个数
    ex_result["micro_avg"]["p_den"] = sum(ex_result[role]["p_den"] for _, role in tag2role.items())#预测的实体个数
    ex_result["micro_avg"]["r_num"] = sum(ex_result[role]["r_num"] for _, role in tag2role.items())#黄金标注的样本中预测正确的实体个数
    ex_result["micro_avg"]["r_den"] = sum(ex_result[role]["r_den"] for _, role in tag2role.items())#黄金标注的实体个数

    for key in all_keys:
        ex_result[key]["p"] = 0 if ex_result[key]["p_num"] == 0 else ex_result[key]["p_num"] / float(
            ex_result[key]["p_den"])
        ex_result[key]["r"] = 0 if ex_result[key]["r_num"] == 0 else ex_result[key]["r_num"] / float(
            ex_result[key]["r_den"])
        ex_result[key]["f1"] = f1(ex_result[key]["p_num"], ex_result[key]["p_den"], ex_result[key]["r_num"],
                                  ex_result[key]["r_den"])

    return ex_result


def eval_tf(preds, golds, docids=[]):
    # normalization mention strings
    for docid in preds:
        for idx_temp in range(len(preds[docid])):
            for role in preds[docid][idx_temp]:
                if role == "incident_type": continue
                for idx in range(len(preds[docid][idx_temp][role])):
                    for idy in range(len(preds[docid][idx_temp][role][idx])):
                        preds[docid][idx_temp][role][idx][idy] = normalize_string(
                            preds[docid][idx_temp][role][idx][idy])
    for docid in golds:
        for idx_temp in range(len(golds[docid])):
            for role in golds[docid][idx_temp]:
                if role == "incident_type": continue
                for idx in range(len(golds[docid][idx_temp][role])):
                    for idy in range(len(golds[docid][idx_temp][role][idx])):
                        golds[docid][idx_temp][role][idx][idy] = normalize_string(
                            golds[docid][idx_temp][role][idx][idy])

    # get eval results
    result = OrderedDict()
    all_keys = list(role for _, role in tag2role.items()) + ["micro_avg", "target"]
    for key in all_keys:
        result[key] = {"p_num": 0, "p_den": 0, "r_num": 0, "r_den": 0, "p": 0, "r": 0, "f1": 0}
    docids = []
    if not docids:
        for docid in preds:
            docids.append(docid)

    for docid in docids:
        pred = preds[docid]
        gold = golds[docid]
        K, V = list(range(len(pred))), list(range(len(gold)))#
        if len(pred) <= len(gold):
            init_maps = [dict(zip(K, p)) for p in itertools.permutations(V, len(K))]
        else:
            init_maps = [dict(zip(p,V)) for p in itertools.permutations(K, len(V))]

        ex_best = None
        map_best = None
        for mapping in init_maps:
            if not is_valid_mapping(mapping):#
                continue
            ex_result = score(mapping, pred, gold)#mapping={0: 0}
            if ex_best is None:
                ex_best = ex_result
                map_best = mapping
            elif ex_result["micro_avg"]["f1"] > ex_best["micro_avg"]["f1"]:
                ex_best = ex_result
                map_best = mapping

        # sum for one docid
        for role in all_keys:
            if role == "micro_avg": continue
            result[role]["p_num"] += ex_best[role]["p_num"]
            result[role]["p_den"] += ex_best[role]["p_den"]
            result[role]["r_num"] += ex_best[role]["r_num"]
            result[role]["r_den"] += ex_best[role]["r_den"]

    # micro average
    result["micro_avg"]["p_num"] = sum(result[role]["p_num"] for _, role in tag2role.items())
    result["micro_avg"]["p_den"] = sum(result[role]["p_den"] for _, role in tag2role.items())
    result["micro_avg"]["r_num"] = sum(result[role]["r_num"] for _, role in tag2role.items())
    result["micro_avg"]["r_den"] = sum(result[role]["r_den"] for _, role in tag2role.items())

    for key in all_keys:
        result[key]["p"] = 0 if result[key]["p_num"] == 0 else result[key]["p_num"] / float(result[key]["p_den"])
        result[key]["r"] = 0 if result[key]["r_num"] == 0 else result[key]["r_num"] / float(result[key]["r_den"])
        result[key]["f1"] = f1(result[key]["p_num"], result[key]["p_den"], result[key]["r_num"], result[key]["r_den"])

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", default="preds_gtt_ree_one.out", type=str, required=False, help="preds output file")
    parser.add_argument("--gold_file", default="./data/test.json", type=str, required=False,help="gold file")
    parser.add_argument("--event_n", default=-1, type=str, required=False, help="event n")
    args = parser.parse_args()

    ## get pred and gold extracts
    preds = OrderedDict()
    golds = OrderedDict()
    with open(args.pred_file, encoding="utf-8") as f:
        out_dict = json.load(f)
        for docid in out_dict:
            doc_id = str(int(docid.split("-")[0][-1]) * 10000 + int(docid.split("-")[2]))
            preds[doc_id] = []
        for docid in out_dict:
            doc_id = str(int(docid.split("-")[0][-1]) * 10000 + int(docid.split("-")[2]))
            preds[doc_id] = out_dict[docid]["pred_templates"]

    with open(args.gold_file, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            docid = str(int(line["docid"].split("-")[0][-1]) * 10000 + int(line["docid"].split("-")[-1]))
            templates_raw = line["templates"]
            templates = []
            for template_raw in templates_raw:
                template = OrderedDict()
                for role, value in template_raw.items():
                    if role == "incident_type":
                        template[role] = value
                    else:
                        template[role] = []
                        for entity_raw in value:
                            entity = []
                            for mention_offset_pair in entity_raw:
                                entity.append(mention_offset_pair[0])
                            if entity:
                                template[role].append(entity)
                if template not in templates:
                    templates.append(template)
            golds[docid] = templates

    with open("./docids_event_n.json", encoding="utf-8") as f:
        docids_event_n = json.load(f)

    if args.event_n == "1,2,3,4":
        all_keys = ["micro_avg"]
        str_print = []
        for num in [1, 2, 3, 4]:
            docids = docids_event_n[str(num)]
            results = eval_tf(preds, golds, docids)
            for key in all_keys:
                str_print += [results[key]["f1"] * 100]
        str_print = ["{:.2f}".format(r) for r in str_print]
        print("print: {}".format(" ".join(str_print)))

    elif args.event_n == ">=2":
        all_keys = ["micro_avg"]
        docids = docids_event_n[args.event_n]
        results = eval_tf(preds, golds, docids)
        str_print = []
        for key in all_keys:
            if key == "micro_avg":
                print("***************** {} *****************".format(key))
            else:
                print("================= {} =================".format(key))

            str_print += [results[key]["p"] * 100, results[key]["r"] * 100, results[key]["f1"] * 100]
            print("P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results[key]["p"] * 100, results[key]["r"] * 100,
                                                                results[key]["f1"] * 100))  # phi_strict
        str_print = ["{:.2f}".format(r) for r in str_print]
        print("print: {}".format(" ".join(str_print)))
        print()


    else:  # all
        all_keys = list(role for _, role in tag2role.items()) + ["micro_avg"]
        docids = []
        results = eval_tf(preds, golds, docids)
        str_print = []
        for key in all_keys:
            if key == "micro_avg":
                print("***************** {} *****************".format(key))
            else:
                print("================= {} =================".format(key))

            str_print += [results[key]["p"] * 100, results[key]["r"] * 100, results[key]["f1"] * 100]
            print("P: {:.2f}%,  R: {:.2f}%, F1: {:.2f}%".format(results[key]["p"] * 100, results[key]["r"] * 100,
                                                                results[key]["f1"] * 100))  # phi_strict
        str_print = ["{:.2f}".format(r) for r in str_print]
        print("print: {}".format(" ".join(str_print)))
        print()





