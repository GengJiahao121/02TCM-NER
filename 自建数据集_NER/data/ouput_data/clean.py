import os
import json
from tqdm import tqdm
import ast
import re

# 清洗result.json

result_file_path = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/result3.json'
cleaned_reault_file_path = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/cleaned_result.json'

with open(result_file_path, 'r') as file:
    result = json.load(file)


def delete_sample(error_list, dataset):
    for sample_index in error_list:
        cleaned_result = [item for item in dataset if item.get('id') != sample_index ]
        print("删除id = {}这条处方....".format(sample_index))
    return cleaned_result

def clean_herb_list(herb_list, id):
    temp_herb_list = []
    # 1. 中草药名称中有的带有括号，去掉括号及括号中的内容
    for herb in herb_list:
        '''
        if isinstance(herb, str):
            pass
        else:
            print(id, herb)
        '''
        #print(id, herb)
        cleaned_herb = re.sub(r'\([^)]*\)', '', herb) # 英文括号
        if cleaned_herb != herb:
            print('herb', id, herb)
        cleaned_cleaned_herb = re.sub(r'\（[^)]*\）', '', cleaned_herb) # 中文括号
        if cleaned_herb != cleaned_cleaned_herb:
            print('herb', id, herb)

        # 2. 
        # 使用正则表达式匹配除中文外的字符
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        if bool(pattern.search(cleaned_cleaned_herb)):
            print('herb', id, cleaned_cleaned_herb)
            continue

        # 3. 非中草药名称，这个只能等词典确定后再匹配了
        # 4. 加减化裁中会含有'去、减、裁'字样标识去掉这味中药，不是增加，在数据集的处理时应注意

        temp_herb_list.append(cleaned_cleaned_herb)
    return temp_herb_list

def clean_individual_characteristics(individual_characteristics_list, id):
    temp_individual_characteristics = []
    for characteristic in individual_characteristics_list:
        # 去掉括号及括号中的内容
        cleaned_characteristic = re.sub(r'\([^)]*\)', '', characteristic) # 英文括号
        if cleaned_characteristic != characteristic:
            print('characteristic', id, characteristic)
        cleaned_cleaned_characteristic = re.sub(r'\（[^)]*\）', '', cleaned_characteristic) # 中文括号
        if cleaned_characteristic != cleaned_cleaned_characteristic:
            print('characteristic', id, characteristic)
        
        # # 使用正则表达式匹配除中文外的字符
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        if bool(pattern.search(cleaned_cleaned_characteristic)):
            print('characteristic', id, cleaned_cleaned_characteristic)
            continue
        
        temp_individual_characteristics.append(cleaned_cleaned_characteristic)
    return temp_individual_characteristics

def clean_syndrome_list(syndrome_list, id):
    # "syndrome": ["脾胃气虚证"]
    

    return syndrome_list

def clean_symptoms_list(symptoms_list, id):
    temp_symptoms_list = []
    # 1. 中草药名称中有的带有括号，去掉括号及括号中的内容
    for symptom in symptoms_list:

        #print(id, symptom)
        cleaned_symptom = re.sub(r'\([^)]*\)', '', symptom) # 英文括号
        if cleaned_symptom != symptom:
            print('symptom', id, symptom)
        cleaned_cleaned_symptom = re.sub(r'\（[^)]*\）', '', cleaned_symptom) # 中文括号
        if cleaned_symptom != cleaned_cleaned_symptom:
            print('symptom', id, symptom)

        # 2. 
        # 使用正则表达式匹配除中文外的字符
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        if bool(pattern.search(cleaned_cleaned_symptom)):
            print('symptom', id, cleaned_cleaned_symptom)
            continue

        # 3. 非中草药名称，这个只能等词典确定后再匹配了
        # 4. 加减化裁中会含有'去、减、裁'字样标识去掉这味中药，不是增加，在数据集的处理时应注意

        temp_symptoms_list.append(cleaned_cleaned_symptom)
    return temp_symptoms_list

# herb_list:
error_herb_list = []
for sample in result:
    herb_list = sample['herb_list']     # ["人参","白术","茯苓","甘草"]
    temp_herb_list = clean_herb_list(herb_list, sample['id'])
    if temp_herb_list:
        sample['herb_list'] = temp_herb_list   
    else:
        # 如果herb_list为空，那么这条sample没有意义，删除这条
        error_herb_list.append(sample['id'])  
print('error_herb_list = ', error_herb_list)
cleaned_result = delete_sample(error_herb_list, result)

# add_or_sub_list:
error_add_or_sub_list = []
for sample in cleaned_result:
    if 'add_or_sub_list' in sample:
        add_or_sub_list = sample['add_or_sub_list'] 
        # [{"individual_characteristics": ["斑疹隐伏不出"], "herbs": ["升麻","葛根","芫荽"]},{"individual_characteristics": ["咽喉肿痛溃烂"],"herbs": ["桔根","牛蒡子","连翘","生甘草"]}]
        temp_add_or_sub_list = []
        for add_or_sub in add_or_sub_list:
            # 清洗个体特征列表
            temp_individual_characteristics = clean_individual_characteristics(add_or_sub['individual_characteristics'], sample['id'])
            # 清洗中药列表
            temp_herbs = clean_herb_list(add_or_sub['herbs'], sample['id'])
            if temp_herbs and temp_individual_characteristics:
                temp_add_or_sub = {}
                temp_add_or_sub['individual_characteristics'] = temp_individual_characteristics
                temp_add_or_sub['herbs'] = temp_herbs
                temp_add_or_sub_list.append(temp_add_or_sub)
        if temp_add_or_sub_list:
            sample['add_or_sub_list'] = temp_add_or_sub_list
        else:
            # 清洗后的加减化裁列表为空，删除add_or_sub_list键值
            del sample['add_or_sub_list']
            print('清洗后的加减化裁列表为空，删除add_or_sub_list键值, sample_id = ', sample['id'])
cleaned_result = result


# syndrome_and_symptoms_list:
error_syndrome_and_symptoms_list = []
for sample in cleaned_result:
    if 'syndrome_and_symptoms_list' in sample:
        syndrome_and_symptoms_list = sample['syndrome_and_symptoms_list'] 
        # [{"syndrome": ["脾胃气虚证"],"symptoms": ["面色萎黄","语声低微","气短乏力","食少便溏","舌淡苔白","脉虚弱"]}]
        temp_syndrome_and_symptoms_list = []
        for syndrome_and_symptoms in syndrome_and_symptoms_list:
            syndrome_list = syndrome_and_symptoms['syndrome']
            syndrome_list = clean_syndrome_list(syndrome_list, sample['id'])
            

            symptoms_list = syndrome_and_symptoms['symptoms']
            symptoms_list = clean_symptoms_list(symptoms_list, sample['id'])

            if symptoms_list:
                syndrome_and_symptoms['syndrome'] = syndrome_list
                syndrome_and_symptoms['symptoms'] = symptoms_list
                temp_syndrome_and_symptoms_list.append(syndrome_and_symptoms)
            else:
                error_syndrome_and_symptoms_list.append(sample['id'])
    else:
        error_syndrome_and_symptoms_list.append(sample['id'])
print('error_syndrome_and_symptoms_list = ', error_syndrome_and_symptoms_list)
cleaned_result = delete_sample(error_syndrome_and_symptoms_list, cleaned_result)

print('*********剩余数据集大小：{}'.format(len(cleaned_result)))

with open(cleaned_reault_file_path, 'w', encoding='utf-8') as file:
    json.dump(cleaned_result, file, ensure_ascii=False, indent=2)   





