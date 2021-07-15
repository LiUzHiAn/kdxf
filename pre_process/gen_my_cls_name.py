import os
import json

with open("../data/kdxf_cls/cls_id_to_name.txt", 'r') as fp:
    names = fp.readlines()

cls_desc = [name for name in names if name != "\n"]
cls_names = [name.split("-")[1] for name in cls_desc]

cls_id_to_name = {}
for cls_id, cls_name in enumerate(cls_names):
    cls_id_to_name[str(cls_id)] = cls_name

with open("../data/kdxf_cls/cls_id_to_name.json", 'w') as fp:
    json.dump(cls_id_to_name, fp, ensure_ascii=False)
