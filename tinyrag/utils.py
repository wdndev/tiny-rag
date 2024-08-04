import os
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
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

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

def write_list_to_txt(data_list, file_path):
    """
    将数据列表每一项作为一行文本保存到文件中。
    
    :param data_list: 要保存的数据列表，每个元素为字符串。
    :param file_path: 输出文件的路径。
    """
    with open(file_path, 'w', encoding='utf-8') as writer:
        for line in data_list:
            writer.write(line + '\n')


def read_file(input_path):
    """
    根据文件扩展名选择合适的读取方法来读取文件内容。
    
    :param input_path: 输入文件的路径。
    :return: 文件内容列表。
    """
    _, file_extension = os.path.splitext(input_path)
    if '.jsonl' == file_extension:
        raw_list = read_jsonl_to_list(input_path)
    elif ".json" == file_extension:
         raw_list = read_json_to_list(input_path)
    elif ".txt" == file_extension :
        raw_list = read_txt_to_list(input_path)
    print(f'{input_path} 已处理完成...')
    return raw_list


def write_file(raw_list: list, output_path):
    """
    根据指定的格式保存数据到文件。
    
    :param format_type: 保存文件的格式，可以是'jsonl'或'txt'。
    :param data_to_save: 待保存的数据列表。
    :param output_path: 输出文件的路径。
    """
    _, file_extension = os.path.splitext(output_path)
    if file_extension == '.jsonl':
        write_list_to_jsonl(raw_list, output_path)
    elif file_extension == ".json":
        write_list_to_json(raw_list, output_path)
    elif file_extension == '.txt':
        write_list_to_txt(raw_list, output_path)
    
    print(f'成功保存至 {output_path}')


def record_log(file_path: str, text: str):
    """
    记录日志到指定文件，如果文件不存在则创建新文件，如果文件已存在则在末尾追加内容。
    
    :param file_path: 日志文件的路径。
    :param text: 要记录的日志文本。
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        # 文件不存在，创建新文件并写入文本
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)
    else:
        # 文件存在，打开文件并在末尾追加文本
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(text)
