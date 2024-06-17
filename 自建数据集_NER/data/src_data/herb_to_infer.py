import os
import json

prescriptions_file_path = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/src_data/prescriptions.json'

herb_infer_file_path = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data'

def process_herb(prescriptions_file_path):
    # 读取 JSON 文件
    with open(prescriptions_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 预处理
    herb_set_list = []
    for prescription in data:
        herb_set = {}
        if 'index' in prescription:  
            herb_set['index'] = prescription['index']
        if '组成' in prescription:
            herb_set['组成'] = prescription['组成']
        if '加减化裁' in prescription:
            herb_set['加减化裁'] = prescription['加减化裁']
        herb_set_list.append(herb_set)
    
    return herb_set_list

if __name__ == '__main__':

    # 1、处理prescriptions.json，将其中的组成字段提取出来
    herb_dataset = process_herb(prescriptions_file_path)

    # 2、问题模版
    # template_content = "给出含有中草药序列如下: \"{herb_sequence}\"。要求只识别所有的中草药实体词；输出格式为列表；请问输出的结果列表为？"

    result = []
    for sample in herb_dataset:
        question_answer_pair = {}
        question_answer_pair['index'] = sample['index']
        question_answer_pair['herb_input'] = sample['组成']
        if '加减化裁' in sample:
            question_answer_pair['ADDorSUB_input'] = sample['加减化裁']
        question_answer_pair['input'] = ""
        question_answer_pair['herb_output'] = ""
        question_answer_pair['ADDorSUB_output'] = ""
        result.append(question_answer_pair)
    
    with open(herb_infer_file_path + '/herb_infer.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("转换成功！！！")



    