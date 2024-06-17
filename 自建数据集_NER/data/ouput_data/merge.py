import os
import json
from tqdm import tqdm
import ast

herb_path = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/herb_infer_output1.json'
'''
{
    "index": 16441,
    "herb_input": "干姜、人参各一两，川芎、甘草（炙）、苦梗（去芦）、厚朴（去粗皮，姜汁制）、白术、陈皮（洗，去白）、白芷、麻黄（去节）各四两，干葛（去粗皮）三两半。",
    "ADDorSUB_input": "如伤风感冷，头疼腰重，咳嗽鼻塞，加葱白煎。",
    "input": "",
    "herb_output": "",
    "ADDorSUB_output": "",
    "herb_ouput": "['干姜', '人参', '川芎', '甘草', '苦梗', '厚朴', '白术', '陈皮', '白芷', '麻黄', '干葛']"
} 
'''
add_or_sub_path = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/加减化裁_infer_output2.json'
'''
{
    "index": 1,
    "herb_input": "人参去芦，白术、茯苓去皮（各9g），甘草炙（6g）。",
    "ADDorSUB_input": "若呕吐，加半夏以降逆止呕；胸膈痞满者，加枳壳、陈皮以行气宽胸；心悸失眠者，加酸枣仁以宁心安神；若畏寒肢冷，脘腹疼痛者，加干姜、附子以温中祛寒。烦渴，加黄芪；胃冷，呕吐涎味，加丁香；呕逆，加藿香；脾胃不和，倍加白术、姜、枣；脾困，加人参、木香、缩砂仁；脾弱腹胀，不思饮食，加扁豆、粟米；伤食，加炒神曲；胸满喘急，加白豆蔻。",
    "input": "",
    "herb_output": "",
    "ADDorSUB_output": [...]
}
'''
symptoms_data1_1_path = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data1_1.json'
symptoms_data1_2_path = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data1_2.json'
symptoms_data2_path = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data2.json'
symptoms_data3_path = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data3.json'
symptoms_data4_path = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data4.json'
'''
{
    "index": 23,
    "fang": "小柴胡汤",
    "symptoms_input": "1、伤寒少阳病证。邪在半表半里，症见往来寒热，胸胁苦满，默默不欲饮食，心烦喜呕，口苦，咽干，目眩，舌苔薄白，脉弦者。2、妇人伤寒，热入血室。经水适断，寒热发作有时。3、疟疾，黄疸等内伤杂病而见以上少阳病证者。",
    "symptoms_output": [...]    
}
'''

# 理想的合并后的.josn文件：
'''
result.json

id: 索引

prescription_name: 方名

herb_sequence: 中草药序列
herb_list: 中草药列表

syndrome_and_symptoms_sequence：证型与症状集合序列
syndrome_and_symptoms_list：证型与症状集合列表
    [
        {
            "syndrome": [],
            "symptoms": []
        },
        {}...
    ]

add_or_sub_sequence: 加减化裁序列
add_or_sub_list:加减化裁列表
    [
        {
            "symptoms": [],
            "herbs": []
        },
        {}...
    ]
'''

result_path = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/result3.json'
if __name__ == '__main__':
    if os.path.exists(result_path):
        # 读取原有的 JSON 数据
        with open(result_path, 'r', encoding='utf-8') as file:
            result = json.load(file)
    else:
        result = []

    with open(herb_path, 'r') as file:
        herb = json.load(file)
    with open(add_or_sub_path, 'r') as file:
        add_or_sub = json.load(file)
    with open(symptoms_data1_1_path, 'r') as file:
        symptoms_data1_1 = json.load(file)
    with open(symptoms_data1_2_path, 'r') as file:
        symptoms_data1_2 = json.load(file)
    with open(symptoms_data2_path, 'r') as file:
        symptoms_data2 = json.load(file)
    with open(symptoms_data3_path, 'r') as file:
        symptoms_data3 = json.load(file)   
    with open(symptoms_data4_path, 'r') as file:
        symptoms_data4 = json.load(file) 

    ## 注意这些是顺序执行的

    # index
    for herb_sample in tqdm(herb, desc='Processing index: '):
        sample_dict = {}
        sample_dict['id'] = herb_sample['index']
        result.append(sample_dict)
    


    # 方名
    for sample_dict in tqdm(result, desc='Processing 方名: '):
        sample_index = sample_dict['id']

        for symptoms_data in  [symptoms_data1_1, symptoms_data1_2, symptoms_data2, symptoms_data3, symptoms_data4]:
            for symptoms_sample in symptoms_data:
                symptoms_sample_index = symptoms_sample['index']
                if sample_index == symptoms_sample_index and 'fang' in symptoms_sample:
                    sample_dict['prescription_name'] = symptoms_sample['fang']
                else:
                    pass
    
    
    # herb_sequence: 中草药序列
    # herb_list: 中草药列表
    herb_error_sample_list = []
    for sample_dict in tqdm(result, desc='Processing herb: '):
        sample_index = sample_dict['id']
        for herb_sample in herb:
            herb_smaple_index = herb_sample['index']
            if sample_index == herb_smaple_index and 'herb_input' in herb_sample:
                sample_dict['herb_sequence'] = herb_sample['herb_input']
            if sample_index == herb_smaple_index and 'herb_ouput' in herb_sample:
                try:
                    sample_dict['herb_list'] = herb_sample['herb_ouput']
                except (ValueError, SyntaxError) as e:
                    print(f"Error: {e}")
                    print("出错处方id = {}......".format(sample_index))
                    print('出错内容 = ', herb_sample['herb_ouput'])
                    herb_error_sample_list.append(sample_index)
    # 删除error_sample_list中的sample
    for sample_index in herb_error_sample_list:
        result = [item for item in result if item.get('id') != sample_index ]
        print("删除id = {}这条处方....".format(sample_index))

    
    # syndrome_and_symptoms_sequence：证型与症状集合序列
    for sample_dict in tqdm(result, desc='Processing 证型与症状集合序列: '):
        sample_index = sample_dict['id']
        for symptoms_data in  [symptoms_data1_1, symptoms_data1_2, symptoms_data2, symptoms_data3, symptoms_data4]:
            for symptoms_sample in symptoms_data:
                symptoms_sample_index = symptoms_sample['index']
                if sample_index == symptoms_sample_index and 'symptoms_input' in symptoms_sample:
                    sample_dict['syndrome_and_symptoms_sequence'] = symptoms_sample['symptoms_input']

    # syndrome_and_symptoms_list：证型与症状集合列表
    '''
    [
        {
            "syndrome": [],
            "symptoms": []
        },
        {}...
    ]
    '''
    ## [symptoms_data1_1, symptoms_data1_2, symptoms_data2, symptoms_data3, symptoms_data4]
    ## symptoms_data1_1, 多组、只有症状集合
    error_list_symptoms_data1_1 = []
    for sample_dict in tqdm(result, desc='Processing symptoms_data1_1, 多组、只有症状集合: '):
        sample_index = sample_dict['id']
        for symptoms_sample in symptoms_data1_1:
            symptoms_sample_index = symptoms_sample['index']
            if sample_index == symptoms_sample_index and 'symptoms_ouput' in symptoms_sample:
                symptoms_ouput = symptoms_sample['symptoms_ouput']
                syndrome_and_symptoms_list = []
                for symptoms_list in symptoms_ouput:
                    syndrome_and_symptoms_sample = {}
                    try:
                        syndrome_and_symptoms_sample['syndrome'] = []
                        syndrome_and_symptoms_sample['symptoms'] = ast.literal_eval(symptoms_list)
                        syndrome_and_symptoms_list.append(syndrome_and_symptoms_sample)
                    except (ValueError, SyntaxError, IndexError) as e:
                        print('error : {}'.format(e))
                        print("error_sample_index : ", sample_index)
                if syndrome_and_symptoms_list:
                    sample_dict['syndrome_and_symptoms_list'] = syndrome_and_symptoms_list
                else:
                    error_list_symptoms_data1_1.append(sample_index)
     # 删除error_sample_list中的sample
    for sample_index in error_list_symptoms_data1_1:
        result = [item for item in result if item.get('id') != sample_index ]
        print("删除id = {}这条处方....".format(sample_index))

    ## symptoms_data1_2, 多组、有证型、有症状集合
    error_list_symptoms_data1_2 = []
    for sample_dict in tqdm(result, desc='Processing symptoms_data1_2, 多组、有证型、有症状集合: '):
        sample_index = sample_dict['id']
        for symptoms_sample in symptoms_data1_2:
            symptoms_sample_index = symptoms_sample['index']
            if sample_index == symptoms_sample_index and 'symptoms_ouput' in symptoms_sample:
                symptoms_ouput = symptoms_sample['symptoms_ouput']
                syndrome_and_symptoms_list = []
                for symptoms_list in symptoms_ouput:
                    syndrome_and_symptoms_sample = {}
                    syndrome_and_symptoms_sample['syndrome'] = symptoms_list[0]
                    syndrome_and_symptoms_sample['symptoms'] = symptoms_list[1]
                    syndrome_and_symptoms_list.append(syndrome_and_symptoms_sample)
                if syndrome_and_symptoms_list:
                    sample_dict['syndrome_and_symptoms_list'] = syndrome_and_symptoms_list
                else:
                    error_list_symptoms_data1_2.append(sample_index)
     # 删除error_sample_list中的sample
    for sample_index in error_list_symptoms_data1_2:
        result = [item for item in result if item.get('id') != sample_index ]
        print("删除id = {}这条处方....".format(sample_index))

    ## symptoms_data2, 一组、只有症状集合
    error_list_symptoms_data2 = []
    for sample_dict in tqdm(result, desc='Processing symptoms_data2, 一组、只有症状集合: '):
        sample_index = sample_dict['id']
        for symptoms_sample in symptoms_data2:
            symptoms_sample_index = symptoms_sample['index']
            if sample_index == symptoms_sample_index and 'symptoms_ouput' in symptoms_sample:
                symptoms_ouput = symptoms_sample['symptoms_ouput']
                syndrome_and_symptoms_list = []
                syndrome_and_symptoms_sample = {}
                try:
                    syndrome_and_symptoms_sample['syndrome'] = []
                    syndrome_and_symptoms_sample['symptoms'] = ast.literal_eval(symptoms_ouput)
                except (ValueError, SyntaxError) as e:
                    print(f"Error: {e}")
                    print("出错处方id = {}......".format(sample_index))
                    error_list_symptoms_data2.append(sample_index)
                syndrome_and_symptoms_list.append(syndrome_and_symptoms_sample)
                if syndrome_and_symptoms_list:
                    sample_dict['syndrome_and_symptoms_list'] = syndrome_and_symptoms_list
                else:
                    error_list_symptoms_data2.append(sample_index)
     # 删除error_sample_list中的sample
    for sample_index in error_list_symptoms_data2:
        result = [item for item in result if item.get('id') != sample_index ]
        print("删除id = {}这条处方....".format(sample_index))

    ## symptoms_data3, 一组、有证型、有症状集合
    error_list_symptoms_data3 = []
    for sample_dict in tqdm(result, desc='Processing symptoms_data3, 一组、有证型、有症状集合: '):
        sample_index = sample_dict['id']
        for symptoms_sample in symptoms_data3:
            symptoms_sample_index = symptoms_sample['index']
            if sample_index == symptoms_sample_index and 'symptoms_ouput' in symptoms_sample:
                symptoms_ouput = symptoms_sample['symptoms_ouput']
                syndrome_and_symptoms_list = []
                syndrome_and_symptoms_sample = {}
                try:
                    symptoms_output = ast.literal_eval(symptoms_ouput)
                    syndrome_and_symptoms_sample['syndrome'] = symptoms_output[0]
                    syndrome_and_symptoms_sample['symptoms'] = symptoms_output[1]
                except (ValueError, SyntaxError, IndexError) as e:
                    print(f"Error: {e}")
                    print("出错处方id = {}......".format(sample_index))
                    error_list_symptoms_data3.append(sample_index)
                syndrome_and_symptoms_list.append(syndrome_and_symptoms_sample)
                if syndrome_and_symptoms_list:
                    sample_dict['syndrome_and_symptoms_list'] = syndrome_and_symptoms_list
                else:
                    error_list_symptoms_data3.append(sample_index)
     # 删除error_sample_list中的sample
    for sample_index in error_list_symptoms_data3:
        result = [item for item in result if item.get('id') != sample_index ]
        print("删除id = {}这条处方....".format(sample_index))


    ## symptoms_data4, 无规律、不处理、待删除
    error_list_symptoms_data4 = []
    for sample_dict in tqdm(result, desc='Processing symptoms_data4, 无规律、不处理、待删除: '):
        sample_index = sample_dict['id']
        for symptoms_sample in symptoms_data4:
            symptoms_sample_index = symptoms_sample['index']
            if sample_index == symptoms_sample_index:
                error_list_symptoms_data4.append(sample_index)
    ### 删除error_sample_list中的sample
    for sample_index in error_list_symptoms_data4:
        result = [item for item in result if item.get('id') != sample_index ]
        print("删除id = {}这条处方....".format(sample_index))

    # add_or_sub_sequence: 加减化裁序列
    # add_or_sub_list:加减化裁列表
    '''
    [
        {
            "symptoms": [],
            "herbs": []
        },
        {}...
    ]
    '''
    ## add_or_sub_sequence: 加减化裁序列
    for sample_dict in tqdm(result, desc='Processing add_or_sub_sequence: 加减化裁序列 '):
        sample_index = sample_dict['id']
        for herb_sample in herb:
            herb_smaple_index = herb_sample['index']
            if sample_index == herb_smaple_index and 'herb_input' in herb_sample:
                sample_dict['herb_sequence'] = herb_sample['herb_input']

    error_list_add_or_sub = []
    for sample_dict in tqdm(result, desc='Processing add_or_sub: '):
        sample_index = sample_dict['id']
        for add_or_sub_dict in add_or_sub:
            add_or_sub_sample_index = add_or_sub_dict['index']
            if sample_index == add_or_sub_sample_index and 'ADDorSUB_input' in add_or_sub_dict:
                sample_dict['add_or_sub_sequence'] = add_or_sub_dict['ADDorSUB_input']
            if sample_index == add_or_sub_sample_index and 'ADDorSUB_output' in add_or_sub_dict:
                add_or_sub_list = []
                ADDorSUB_output = add_or_sub_dict['ADDorSUB_output']
                for add_or_sub_sub_list in ADDorSUB_output:
                    add_or_sub_sample = {}
                    try:
                        if len(add_or_sub_sub_list) == 2:
                            add_or_sub_sample['individual_characteristics'] = add_or_sub_sub_list[0]
                            add_or_sub_sample['herbs'] = add_or_sub_sub_list[1]
                            add_or_sub_list.append(add_or_sub_sample)
                    except (ValueError, SyntaxError, IndexError) as e:
                        print(f"Error: {e}")
                        print("处方id = {} 有错误，但不代表所有这个sample全部错误......".format(sample_index))
                if add_or_sub_list:
                    sample_dict['add_or_sub_list'] = add_or_sub_list
                else:
                    print("出错处方id = {}， 这个sample中的加减化裁全部不可用！......".format(sample_index))
                    error_list_add_or_sub.append(sample_index)
    ### 如果加减化裁不可用，不代表这个sample不可用，所以不删除
    for sample_index in error_list_add_or_sub:
        # result = [item for item in result if item.get('id') != sample_index ]
        print("id = {}这条处方，原本有加减化裁，但是不可用！....".format(sample_index))     

    print('*********剩余数据集大小：{}  ***************'.format(len(result)))

    # 最后统一保存文件
    with open(result_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)










