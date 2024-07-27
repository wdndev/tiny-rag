import os
import json
from file_helpers import *


def jsonl_to_text(json_path, txt_path):
    jsonl_list = read_json_to_list(json_path)
    txt_list = []
    for item in jsonl_list:
        content = item["completion"]
        txt_list.append(content)
    write_list_to_txt(txt_list, txt_path)

def main():
    json_path = "data/raw_data/wikipedia-cn-20230720-filtered.json"
    txt_path = "data/raw_data/wikipedia-cn-20230720-filtered.txt"
    jsonl_to_text(json_path, txt_path)

if __name__ == "__main__":
    main()
