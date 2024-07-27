import json

def read_jsonl_to_list(file_path):
    """
    从jsonl文件读取数据到列表中。
    
    参数:
    file_path (str): jsonl文件的路径
    
    返回:
    list: 包含jsonl文件中数据的列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def write_list_to_jsonl(data, file_path):
    """
    将列表数据写入jsonl文件。
    
    参数:
    data (list): 需要写入jsonl文件的列表数据
    file_path (str): 目标jsonl文件的路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def write_list_to_json(data, file_path):
    """
    将列表数据写入JSON文件。

    参数:
    data (list): 需要写入JSON文件的列表数据
    file_path (str): 目标JSON文件的路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

        
def read_json_to_list(file_path):
    """
    从JSON文件读取数据到列表中。

    参数:
    file_path (str): JSON文件的路径

    返回:
    list: 包含JSON文件中数据的列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_list_to_txt(data, file_path):
    """
    将列表数据写入txt文件，每个元素占一行。
    
    参数:
    data (list): 需要写入txt文件的列表数据
    file_path (str): 目标txt文件的路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(str(item) + "\n")

def read_txt_to_list(file_path):
    """
    从txt文件读取数据到列表中，每行作为一个列表元素。
    
    参数:
    file_path (str): txt文件的路径
    
    返回:
    list: 包含txt文件中数据的列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [line.strip() for line in f]
    return data
