from utils import Huozi
import json
import os
import re
from tqdm import tqdm
import ast

'''
处理过程：
1. 先划分为4类情况，再分别处理

input_data1 = [] # 记录有多种情况，且以数字序号开头的数据 input_data1 [{'index': 1, 'fang': xxx, 'symptoms_input': xxxxxxxx, 'symptoms_input_preprocess': [子句1, 子句2, ....], 'symptoms_output': xxx}, {}, ...]
input_data2 = [] # 记录无数字开头的数据，且只有一个句号（可理解为只有症状描述） input_data2 [{'index': 1, 'fang': xxx, 'symptoms_input': xxxxxxxx, 'symptoms_input_preprocess': xxxxxxxxx, 'symptoms_output': xxx}, {}, ...]
input_data3 = [] # 记录无数字开头的数据，且只有二个句号（可理解为前面为证型，后面为证型描述） ..
input_data4 = [] # 记录无数字开头的数据，且有三个及以上的句号（数据不规范，后续处理）.. 

2. 分别处理 

python 正则表达式、数据的清洗去除标点符号空格符(数据分析)

哈工大SCRI的 活字模型 + 提示技术（大模型 + 实体识别）

3. 结果保存：

save_file_path_input_data1_1 = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data1_1.json'
save_file_path_input_data1_2 = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data1_2.json'
save_file_path_input_data2 = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data2.json'
save_file_path_input_data3 = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data3.json'
save_file_path_input_data4 = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data4.json'

'''


precision = "fp16"
huozi2_model_name_or_path = "/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/LLM/huozi-7b-rlhf"
dataset_dir = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/symptoms_infer.json' # 输入
save_file_path_input_data1_1 = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data1_1.json'
save_file_path_input_data1_2 = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data1_2.json'
save_file_path_input_data2 = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data2.json'
save_file_path_input_data3 = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data3.json'
save_file_path_input_data4 = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/symptoms_infer_output_input_data4.json'


model = Huozi(huozi2_model_name_or_path, precision)
generate_kwargs = {
    "max_new_tokens": 512,
    "temperature": 0.001,
    "do_sample": True,
    "repetition_penalty":  1.03,
    "top_k": 40,
    "top_p": 0.01,
}

def add_result_to_file( save_file_path, result):
    if os.path.exists(save_file_path):
        # 读取原有的 JSON 数据
        with open(save_file_path, 'r', encoding='utf-8') as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    # 追加新数据到现有数据中
    existing_data.extend(result)

    with open(save_file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=2)

    # 清空情况列表
    result.clear()

    #logging.info("写入文件成功！已经写入{}个文件！！！\n\n".format(i+1))
    print("已写入{}个文件。\n".format(len(existing_data)))

def main():
    
    with open(dataset_dir, 'r') as f:
        input_data = json.load(f)

    input_data1 = [] # 记录有多种情况，且以数字序号开头的数据
    input_data2 = [] # 记录无数字开头的数据，且只有一个句号（可理解为只有症状描述）
    input_data3 = [] # 记录无数字开头的数据，且只有二个句号（可理解为前面为证型，后面为证型描述）
    input_data4 = [] # 记录无数字开头的数据，且有三个及以上的句号（数据不规范，后续处理）

    
    for data in input_data:
        sample = {}
        if 'index' in data:
            sample['index'] = data['index']
        if 'fang' in data:
            sample['fang'] = data['fang']
        if 'symptoms_input' in data:
            sample['symptoms_input'] = data['symptoms_input']
        sample['symptoms_output'] = ""

        sentences = data['symptoms_input']
        sentences = re.split(r'。|。\s*', sentences.strip())
        # 去除空字符串
        sentences = [sentence.strip() for sentence in sentences if sentence]

        # 得到input_data1 [{'index': 1, 'fang': xxx, 'symptoms_input': xxxxxxxx, 'symptoms_input_preprocess': [子句1, 子句2, ....], 'symptoms_output': xxx}, {}, ...]
        if any(re.match(r'^[①②③④⑤⑥⑦⑧]', sentence) for sentence in sentences) or any(re.match(r'^\d', sentence) for sentence in sentences):
            # 符合input_data1要求
            # 划分子句
            result = []
            current_sentence = ''
            for sentence in sentences:
                if re.match(r'^\d', sentence) or re.match(r'^[①②③④⑤⑥⑦⑧]', sentence):
                    # 新的子句以数字开头，将当前子句添加到结果列表，并重置当前子句
                    if current_sentence:
                        result.append(current_sentence)
                    current_sentence = sentence
                else:
                    if current_sentence:
                        # 同一子句中的其他部分，追加到当前子句
                        current_sentence += '。' + sentence
                    else:
                        current_sentence = sentence

            # 将最后一个子句添加到结果列表
            if current_sentence:
                result.append(current_sentence)

            sample['symptoms_input_preprocess'] = result

            input_data1.append(sample)
    
        # 记录无数字开头的数据
        else:
            periods = re.findall(r'。', data['symptoms_input'])
            count = len(periods)
            sample['symptoms_input_preprocess'] = data['symptoms_input']
            if count == 1:
                input_data2.append(sample)
            elif count == 2:
                input_data3.append(sample)
            else:
                input_data4.append(sample)

    # 进一步处理input_data1,将划分的子句中每个子句都有两个子句的数据和其他再分成两个得到input_data1-1和input_data1-2
    input_data1_1 = []
    input_data1_2 = []
    for data in input_data1:
        symptoms_instr_list = data['symptoms_input_preprocess']
        flag = 1
        for sentence in symptoms_instr_list:
            periods = re.findall(r'。', sentence)
            count = len(periods)
            if count >= 1:
                continue
            else:
                flag = 0
                break
        if flag:
            input_data1_2.append(data) # 每个子句都有两个子句（可理解为每个子句中，前面句子是证型描述，后面句子是症状描述）
        else:
            input_data1_1.append(data) # 其他（只识别症状）

    # *************
    mode = 2
    # *************

    result = []
    history = None
    
    # 处理input_data1_1
    # symptoms_ouput = [ [症状集合], [],  ......]
    if mode == 0:
        background = "我希望你能够以一名中医领域专家的身份帮我完成如下的任务。首先我会给你一段文本，然后你需要基于中医领域常见的症状词尽可能多的提取出这段文本中的症状实体词，构成症状实体词列表。你的输出需要严格按照python列表的格式输出，我接下来会给你个例子。你需要结合下列例子理解我的上述要求，并且按照要求完成任务。"
        example1 = "输入文本为：⑤《鸡峰》：表里俱虚，伤冒寒冷，腹胁胀满，呕逆痰涎；及邪中阴经，手足厥冷，既吐且利，小便频数，里寒，身体疼痛，脉细微，下利清谷，头痛恶寒，亡阳自汗。 正确的输出格式为：['表里俱虚', '伤冒寒冷', '腹胁胀满', '呕逆痰涎', '邪中阴经', '手足厥冷', '吐且利', '小便频数', '里寒', '身体疼痛', '脉细微', '下利清谷', '头痛恶寒', '亡阳自汗']"
        example2 = "" 
        prompt_base = background + example1
        print('prompt_base: ', prompt_base, '\n')
        symptoms_ouput, history= model.chat(generate_kwargs, prompt_base, history=history)
        print('prompt_output: ', symptoms_ouput, '\n')

        while True:
            query = input(">>>> ")
            if query == "q":
                break

            response, history = model.chat(generate_kwargs, query, history=history)
            print(f"Bot: {response}")
        # >>>> 你的输出格式不准确，准确的输出内容中只有列表。
        print("进入识别...")

        # 进入识别之前，把history内容填满
        pre_history = [
            ('①《伤寒论》：风湿相搏，骨节疼烦，掣痛不得屈伸，近之则痛剧，汗出短气，小便不利，恶风不欲去衣，或身微肿者', "['风湿相搏', '骨节疼烦', '掣痛不得屈伸', '痛剧', '汗出', '短气', '小便不利', '恶风', '不欲去衣', '身微肿']"), 
            ('②《外台》引《近效方》：风虚头重眩，苦极不知食味', "['风虚头重眩', '苦极不知食味']"),
            ('1.《伤寒论》：发汗过多，其人叉手自冒心，心下悸，欲得按者', "['发汗过多', '叉手自冒心', '心下悸', '欲得按者']"),
            ('2.《伤寒论今释》引《证治大还》：妇人生产不快，或死腹中', "['妇人生产不快', '或死腹中']"),
            ('①《活幼口议》：脾积寒热，其状如疟，或头痛呕逆，久则二三岁不歇，左胁有块，小者如桃李，大者似杯碟',"['脾积寒热', '状如疟', '头痛呕逆', '久则二三岁不歇', '左胁有块', '小者如桃李', '大者似杯碟']"),
            ('②《古今医统大全》：胃热口气，痰饮呕逆，不思饮食',"['胃热口气', '痰饮呕逆', '不思饮食']"),
            ('①伤寒，医以丸药大下之，身热不去，微烦者，栀子干姜汤主之',"['伤寒', '身热不去', '微烦', '栀子干姜汤']"),
            ('②《圣济总录》：时气病后，余毒不尽上攻，目赤涩痛，或生障翳',"['时气病后', '余毒不尽上攻', '目赤涩痛', '或生障翳']"),
            ('②《伤寒论》：霍乱，恶寒，脉微而复利，利止，亡血也',"['霍乱', '恶寒', '脉微而复利', '利止', '亡血']"),
            ('⑦真阳衰微，元气亦虚之证。四肢厥逆，恶寒蜷卧脉微而复自下利，利虽止而余证仍在者',"['真阳衰微', '元气亦虚之证', '四肢厥逆', '恶寒蜷卧', '脉微而复', '自下利', '利虽止而余证仍在']"),
            ('⑤《鸡峰》：表里俱虚，伤冒寒冷，腹胁胀满，呕逆痰涎；及邪中阴经，手足厥冷，既吐且利，小便频数，里寒，身体疼痛，脉细微，下利清谷，头痛恶寒，亡阳自汗',"['表里俱虚', '伤冒寒冷', '腹胁胀满', '呕逆痰涎', '邪中阴经', '手足厥冷', '吐且利', '小便频数', '里寒', '身体疼痛', '脉细微', '下利清谷', '头痛恶寒', '亡阳自汗']"),
            ('②《景岳全书》：破伤风，邪传于里，舌强口噤，项背反张，筋惕搐搦，痰涎壅盛',"['破伤风', '邪传于里', '舌强口噤', '项背反张', '筋惕搐搦', '痰涎壅盛']"),
            ('①《宣明论》：中外诸热，寝汗咬牙，睡语惊悸，溺血淋闭，咳血衄血，瘦弱头痛，并骨蒸、肺痿、喘嗽',"['中外诸热', '寝汗咬牙', '睡语惊悸', '溺血淋闭', '咳血衄血', '瘦弱头痛', '并骨蒸', '肺痿', '喘嗽']"),
            ('②《杂病源流犀烛》：疹后痨，疹既收没，毒邪犹郁于肌肉间，昼夜发热，渐致发焦肤槁，羸瘦如柴，变成骨蒸痨瘵',"['疹后痨', '疹既收没', '毒邪犹郁于肌肉间', '昼夜发热', '渐致发焦肤槁', '羸瘦如柴', '变成骨蒸痨瘵']"),
            ('①《辨证录》：素多恼怒，容易动气，一旦两胁胀满，发寒发热，既而胁痛之极，手按痛处不可忍',"['素多恼怒', '容易动气', '一旦两胁胀满', '发寒发热', '胁痛之极', '手按痛处不可忍']"),
            ('①《金匮》（附方引《近效方》）：风虚头重眩苦极，不知食味',"['风虚头重眩', '苦极', '不知食味']"),
            ('①《伤寒论》：伤寒，服汤药，下利不止，心下痞粳。服泻心汤己，复以他药下之，利不止。医以理中与之，利益甚，此利在下焦', "['伤寒', '服汤药', '下利不止', '心下痞粳', '服泻心汤已', '复以他药下之', '利不止', '医以理中与之', '利益甚', '此利在下焦']"),
            ('②《证治准绳·类方》：大肠腑发咳，咳而遗矢', "['大肠腑发咳', '咳而遗矢']"),
            ('2.《千金》：咳而大逆，上气胸满，喉中不利，如水鸡声，其脉浮者', "['咳而大逆', '上气胸满', '喉中不利', '如水鸡声', '其脉浮者']"),
            ('2、热伤血络，斑色紫黑、吐血、衄血、便血、尿血等，舌绛红，脉数', "['热伤血络', '斑色紫黑', '吐血', '衄血', '便血', '尿血', '舌绛红', '脉数']"),
            ('3、蓄血瘀热，喜忘如狂，漱水不欲咽，大便色黑易解等', "['蓄血瘀热', '喜忘如狂', '漱水不欲咽', '大便色黑易解']"),
            ('①《千金》引张仲景方：中风，手足拘挛，百节痛烦，烦热心乱，恶寒经日，不欲饮食', "['中风', '手足拘挛', '百节痛烦', '烦热心乱', '恶寒经日', '不欲饮食']"),
            ('①行役劳苦，动作不休，以至筋缩不伸，卧床呻吟，不能举步，遍身疼痛，手臂疫麻', "['行役劳苦', '动作不休', '以至筋缩不伸', '卧床呻吟', '不能举步', '遍身疼痛', '手臂疫麻']"),
            ('肾阴、肾阳不足而虚火上炎之更年期综合征，高血压病，肾炎、肾盂肾炎，尿路感染，闭经', "['肾阴、肾阳不足', '虚火上炎', '更年期综合征', '高血压病', '肾炎', '肾盂肾炎', '尿路感染', '闭经']"),
        ]
        
        history += pre_history
        for i, data in tqdm(input_data1_1, desc="Processing"):
            symptoms_input_preprocess = data['symptoms_input_preprocess']
            #print('sentences_input = ', symptoms_input_preprocess, '\n')
            sentences_ouput = []
            for sentence in symptoms_input_preprocess:
                input_answer = sentence
                #print('sentence = ', input_answer, '\n')
                symptoms_ouput, h= model.chat(generate_kwargs, input_answer, history=history)
                try:
                    symptoms_ouput = ast.literal_eval(symptoms_ouput)
                    #print("Response: ", symptoms_ouput, '\n')
                    sentences_ouput.append(symptoms_ouput)
                except (ValueError, SyntaxError) as e:
                    print(f"Error: {e}")
                    print('******')
            print('sentences_ouput = ', sentences_ouput, '\n')
            #print('*************************************\n')
            data['symptoms_ouput'] = sentences_ouput
            result.append(data)
            # 每完成多少个数据就更新到文件中
            if len(result) == 10:
                add_result_to_file(save_file_path_input_data1_1, result)
        add_result_to_file(save_file_path_input_data1_1, result)
        print('end!')

    # 处理input_daat1_2
    # symptoms_ouput = [ [[证型], [症状集合]], [[...], [...]], [[...], [...]], ......]
    if mode == 1:
        background = "我希望你能够以一名中医领域专家的身份帮我完成如下的任务。首先我会给你一段文本，然后你需要尽可能多的提取出这段文本中的证型实体词和症状实体词，构成证型实体词列表和症状实体词列表。你的输出需要严格按照python列表的格式输出，我接下来会给你几个例子。你需要结合下列例子理解我的上述要求，并且按照要求完成任务。"
        example1 = "输入：脾胃气虚证。面色萎黄，语声低微，气短乏力，食少便溏，舌淡苔白，脉虚弱。输出：[['脾胃气虚证'], ['面色萎黄', '语声低微', '气短乏力', '食少便溏', '舌淡苔白', '脉虚弱']]"
        example2 =  ""
        prompt_base = background + example1
        print('prompt_base: ', prompt_base, '\n')
        symptoms_ouput, history= model.chat(generate_kwargs, prompt_base, history=history)
        print('prompt_output: ', symptoms_ouput, '\n')

        while True:
            query = input(">>>> ")
            if query == "q":
                break

            response, history = model.chat(generate_kwargs, query, history=history)
            print(f"Bot: {response}")
        # >>>> 你的输出格式不正确，正确的输出格式为：[['脾胃气虚证'], ['面色萎黄', '语声低微', '气短乏力', '食少便溏', '舌淡苔白', '脉虚弱']]
        print("进入识别...")

        # 如果有子句中有两个句号
        pre_history = [
            ('风温初起，表热轻证。证见但咳，身热不甚，口微渴，苔薄白，脉浮数。', "[['风温初起', '表热轻证'], ['但咳', '身热不甚', '口微渴', '苔薄白', '脉浮数']]"),
            ('阴虚火旺盗汗。发热盗汗，面赤心烦，口干唇燥，大便干结，小便黄赤，舌红苔黄，脉数。', "[['阴虚火旺盗汗'], ['发热盗汗', '面赤心烦', '口干唇燥', '大便干结', '小便黄赤', '舌红苔黄', '脉数']]"),
            ('肝郁血虚脾弱证。两胁作痛，头痛目眩，口燥咽干，神疲食少，或月经不调，乳房胀痛，脉弦而虚者。', "[['肝郁血虚脾弱证'], ['两胁作痛', '头痛目眩', '口燥咽干', '神疲食少', '或月经不调', '乳房胀痛', '脉弦而虚']]"),
            ('肺热喘咳。气喘咳嗽，皮肤蒸热，日晡尤甚，舌红苔黄，脉细数。', "[['肺热喘咳'], ['气喘咳嗽', '皮肤蒸热', '日晡尤甚', '舌红苔黄', '脉细数']]"),
            ('脏躁。症见精神恍惚，常悲伤欲哭，不能自主，心中烦乱，睡眠不安，甚则言行失常，呵欠频作，舌淡红苔少，脉细微数。', "[['脏躁'], ['精神恍惚', '常悲伤欲哭', '不能自主', '心中烦乱', '睡眠不安', '言行失常', '呵欠频作', '舌淡红苔少', '脉细微数']]"),
            ('痰壅气逆食滞证。咳嗽喘逆，痰多胸痞，食少难消，舌苔白腻，脉滑。', "[['痰壅气逆食滞证'], ['咳嗽喘逆', '痰多胸痞', '食少难消', '舌苔白腻', '脉滑']]"),
            ('湿温初起及暑温夹湿之湿重于热证。头痛恶寒，身重疼痛，肢体倦怠，面色淡黄，胸闷不饥，午后身热，苔白不渴，脉弦细而濡。', "[['湿温初起及暑温夹湿之湿重于热证'], ['头痛', '身重疼痛', '肢体倦怠', '面色淡黄', '胸闷不饥', '午后身热', '苔白不渴', '脉弦细而濡']]"),
            ('肝气郁结证。症见胸膈胀闷，上气喘急，心下痞满，不思饮食，脉弦。', "[['肝气郁结证'], ['胸膈胀闷', '上气喘急', '心下痞满', '不思饮食', '脉弦']]"),
            ('风湿在表之痹证。肩背痛不可回顾，头痛身重，或腰脊疼痛，难以转侧，苔白，脉浮。', "[['风湿在表之痹证'], ['肩背痛不可回顾', '头痛身重', '腰脊疼痛', '难以转侧', '苔白', '脉浮']]"),
            ('风痰上扰证。眩晕，头痛，胸膈痞闷，恶心呕吐，舌苔白腻，脉弦滑。', "[['风痰上扰证'], ['眩晕', '头痛', '胸膈痞闷', '恶心呕吐', '舌苔白腻', '脉弦滑']]"),
            ('外感风寒表实证。恶寒发热，头身疼痛，无汗而喘，舌苔薄白，脉浮紧。', "[['外感风寒表实证'], ['恶寒发热', '头身疼痛', '无汗而喘', '舌苔薄白', '脉浮紧']]"),
            ('外感风寒湿邪，内有蕴热证。恶寒发热，无汗，头痛项强，肢体酸楚疼痛，口苦微渴，舌苔白或微黄，脉浮。', "[['外感风寒湿邪，内有蕴热证'], ['恶寒发热', '无汗', '头痛项强', '肢体酸楚疼痛', '口苦微渴', '舌苔白或微黄', '脉浮']]"),
            ('痹证日久，肝肾两虚，气血不足证。腰膝疼痛、痿软，肢节屈伸不利，或麻木不仁，畏寒喜温，心悸气短，舌淡苔白，脉细弱。', "[['痹证日久，肝肾两虚，气血不足证'], ['腰膝疼痛', '痿软', '肢节屈伸不利', '或麻木不仁', '畏寒喜温', '心悸气短', '舌淡苔白', '脉细弱']]"),
            ('肝血不足，虚热内扰证。虚烦失眠，心悸不安，头目眩晕，咽干口燥，舌红，脉弦细。', "[['肝血不足，虚热内扰证'], ['虚烦失眠', '心悸不安', '头目眩晕', '咽干口燥', '舌红', '脉弦细']]"),
            ('血虚寒凝，瘀血阻滞证。产后恶露不行，小腹冷痛。', "[['血虚寒凝，瘀血阻滞证'], ['产后恶露不行', '小腹冷痛']]"),
            ('心经火热证。心胸烦热，口渴面赤，意欲饮冷，以及口舌生疮；或心热移于小肠，小便赤涩刺痛，舌红，脉数。', "[['心经火热证'], ['心胸烦热', '口渴面赤', '意欲饮冷', '口舌生疮', '或', '心热移于小肠', '小便赤涩刺痛', '舌红', '脉数']]"),
            ('湿热黄疸。一身面目俱黄，黄色鲜明，发热，无汗或但头汗出，口渴欲饮，恶心呕吐，腹微满，小便短赤，大便不爽或秘结，舌红苔黄腻，脉沉数或滑数有力。', "[['湿热黄疸'], ['一身面目俱黄', '黄色鲜明', '发热', '无汗或但头汗出', '口渴欲饮', '恶心呕吐', '腹微满', '小便短赤', '大便不爽或秘结', '舌红苔黄腻', '脉沉数或滑数有力']]"),
            ('肝胃虚寒，浊阴上逆证。食后泛泛欲吐，或呕吐酸水，或干呕，或吐清涎冷沫，胸满脘痛，巅顶头痛，畏寒肢冷，甚则伴手足逆冷，大便泄泻，烦躁不宁，舌淡苔白滑，脉沉弦或迟。', "[['肝胃虚寒，浊阴上逆证'], ['食后泛泛欲吐', '呕吐酸水', '干呕', '吐清涎冷沫', '胸满脘痛', '巅顶头痛', '畏寒肢冷', '手足逆冷', '大便泄泻', '烦躁不宁', '舌淡苔白滑', '脉沉弦或迟']]"),
            ('气血两虚证。面色苍白或萎黄，头晕耳眩，四肢倦怠，气短懒言，心悸怔忡，饮食减少，舌淡苔薄白，脉细弱或虚大无力。', "[['气血两虚证'], ['面色苍白或萎黄', '头晕耳眩', '四肢倦怠', '气短懒言', '心悸怔忡', '饮食减少', '舌淡苔薄白', '脉细弱或虚大无力']]"),
            ('湿痰证。咳嗽痰多，色白易咯，恶心呕吐，胸膈痞闷，肢体困重，或头眩心悸，舌苔白滑或腻，脉滑。', "[['湿痰证'], ['咳嗽痰多', '色白易咯', '恶心呕吐', '胸膈痞闷', '肢体困重', '头眩心悸', '舌苔白滑或腻', '脉滑']]"),
        ]
        
        history = history + pre_history
        print('history = ', history, '\n')

        for data in tqdm(input_data1_2, desc="Processing"):
            symptoms_input_preprocess = data['symptoms_input_preprocess']
            #print('sentences_input = ', symptoms_input_preprocess, '\n')
            sentences_ouput = []
            for sentence in symptoms_input_preprocess:
                input_answer = sentence
                #print('sentence = ', input_answer, '\n')
                symptoms_ouput, h= model.chat(generate_kwargs, input_answer, history=history)
                try:
                    symptoms_ouput = ast.literal_eval(symptoms_ouput)
                    #print("Response: ", symptoms_ouput, '\n')
                    sentences_ouput.append(symptoms_ouput)
                except (ValueError, SyntaxError) as e:
                    print(f"Error: {e}")
                    print('symptoms_ouput = ', symptoms_ouput)
                    print('******')
            #print('sentences_ouput = ', sentences_ouput, '\n')
            #print('*********************************\n')
            data['symptoms_ouput'] = sentences_ouput
            result.append(data)
            # 每完成多少个数据就更新到文件中
            if len(result) == 10:
                add_result_to_file(save_file_path_input_data1_2, result)
        add_result_to_file(save_file_path_input_data1_2, result)
        print('end!')
        

    # 处理input_data2, 无数字开头的数据，且只有一个句号（可理解为只有症状描述）
    if mode == 2:
        background = "我希望你能够以一名中医领域专家的身份帮我完成如下的任务。首先我会给你一段文本，然后你需要基于中医领域常见的症状词尽可能多的提取出这段文本中的症状实体词，构成症状实体词列表。你的输出需要严格按照python列表的格式输出，我接下来会给你个例子。你需要结合下列例子理解我的上述要求，并且按照要求完成任务。"
        example1 = "输入文本为：太阳病，得之八九日，如疟状，发热恶寒，热多寒少，其人不呕，清便欲自可，一日二三度发，面色反有热色，身痒者。 正确的输出格式为：['发热恶寒', '热多寒少', '不呕', '清便欲自可', '一日二三度发', '面色反有热色', '身痒']"
        example2 = "" 
        prompt_base = background + example1
        print('prompt_base: ', prompt_base, '\n')
        symptoms_ouput, history= model.chat(generate_kwargs, prompt_base, history=history)
        print('prompt_output: ', symptoms_ouput, '\n')

        while True:
            query = input(">>>> ")
            if query == "q":
                break

            response, history = model.chat(generate_kwargs, query, history=history)
            print(f"Bot: {response}")
        # >>>> 你的输出格式不准确，准确的输出内容中只有列表。
        print("进入识别...")

        # 进入识别之前，把history内容填满
        pre_history = [
            ('素体阳虚，外感风寒，无汗恶寒，发热蜷卧，苔白，脉反沉者。', "['素体阳虚', '外感风寒', '无汗恶寒', '发热蜷卧', '苔白', '脉反沉']"), 
            ('血痹，肌肤麻木不仁，脉微涩而紧。', "['血痹', '肌肤麻木不仁', '脉微涩而紧']"),
            ('邪火内炽，迫血妄行，吐血、衄血；或湿热内蕴而成黄疸，胸痞烦热；或积热上冲而致目赤肿痛，口舌生疮；或外科疮疡，见有心胸烦热，大便干结者。', "['邪火内炽', '迫血妄行', '吐血', '衄血', '湿热内蕴', '黄疸', '胸痞烦热', '积热上冲', '目赤肿痛', '口舌生疮', '外科疮疡', '心胸烦热', '大便干结']"),
            ('气血亏损，肾寒精冷，肚腹疼痛，腰膝无力。', "['气血亏损', '肾寒精冷', '肚腹疼痛', '腰膝无力']"),
            ('半身不遂，口眼歪斜，手足战掉，语言謇涩，肢体麻痹，神思昏乱，头目眩重，痰诞壅盛，筋脉拘挛，屈伸转侧不便，涕唾不收。', "['半身不遂', '口眼歪斜', '手足战掉', '语言謇涩', '肢体麻痹', '神思昏乱', '头目眩重', '痰诞壅盛', '筋脉拘挛', '屈伸转侧不便', '涕唾不收']"),
            ('太阳病，外证未除，而数下之，遂协热下利，利下不止，心下痞硬，表里不解者。', "['太阳病', '外证未除', '数下', '协热下利', '利下不止', '心下痞硬', '表里不解']"),
            ('外感风寒表实，项背强，无汗恶风，或自下利，或血衄；痉病，气上冲胸，口噤不语，无汗，小便少，或卒倒僵仆。', "['外感风寒表实', '项背强', '无汗恶风', '自下利', '血衄', '痉病', '气上冲胸', '口噤不语', '无汗', '小便少', '卒倒僵仆']"),
            ('一切色欲过度，肾经虚寒缩阳之症。', "['色欲过度', '肾经虚寒', '缩阳之症']"),
            ('阳明温病，无上焦证，数日不大便，当下之，若其人阴素虚，不可行承气者。', "['阳明温病', '无上焦证', '数日不大便', '当下之', '阴素虚', '不可行承气']"),
            ('外感风寒表实，项背强，无汗恶风，或自下利，或血衄；痉病，气上冲胸，口噤不语，无汗，小便少，或卒倒僵仆。', "['外感风寒表实', '项背强', '无汗恶风', '自下利', '血衄', '痉病', '气上冲胸', '口噤不语', '无汗', '小便少', '卒倒僵仆']"),
            ('痹症有淤血者', "['痹症', '淤血']"),
            ('慢惊，脾虚肝旺，风痰盛者。', "['慢惊', '脾虚肝旺', '风痰盛']"),
            ('风水证，症见发热、恶风寒、一身悉肿、口微渴、骨节疼痛；或身体反重而酸、汗自出；或目窠上微拥即眼睑水肿，如蚕新卧起伏、其颈脉动、按手足肿上陷而不起、脉浮或寸口脉沉滑。', "['风水证', '发热', '恶风寒', '一身悉肿', '口微渴', '骨节疼痛', '身体反重', '酸', '汗自出', '目窠上微拥', '眼睑水肿', '蚕新卧起伏', '颈脉动', '按手足肿上陷而不起', '脉浮', '寸口脉沉滑']"),
            ('心阳不足证，烦躁不安，心悸，或失眠，心胸憋闷，畏寒肢冷，气短自汗，面色苍白，舌淡苔白，脉迟无力。', "['心阳不足证', '烦躁不安', '心悸', '失眠', '心胸憋闷', '畏寒肢冷', '气短自汗', '面色苍白', '舌淡苔白', '脉迟无力']"),
            ('阴虚火盛，下焦湿热等证', "['阴虚火盛', '下焦湿热等证']"),
            ('痤痹疮作痒，抓之又疼，坐如糠稳，难以安睡。', "['痤痹疮', '作痒', '抓之又疼', '坐如糠稳', '难以安睡']"),
            ('半身不遂，口眼歪斜，手足战掉，语言謇涩，肢体麻痹，神思昏乱，头目眩重，痰诞壅盛，筋脉拘挛，屈伸转侧不便，涕唾不收。', "['半身不遂', '口眼歪斜', '手足战掉', '语言謇涩', '肢体麻痹', '神思昏乱', '头目眩重', '痰诞壅盛', '筋脉拘挛', '屈伸转侧不便', '涕唾不收']"),
            ('脾胃虚寒，自利不渴，呕吐腹痛，不欲饮食，中寒霍乱，阳虚失血，胸痹虚证，病后喜唾，小儿慢惊。', "['脾胃虚寒', '自利不渴', '呕吐腹痛', '不欲饮食', '中寒霍乱', '阳虚失血', '胸痹虚证', '病后喜唾', '小儿慢惊']"),
            ('气虚肿满，痰饮结聚，脾胃不和，变生诸证者。', "['气虚肿满', '痰饮结聚', '脾胃不和', '变生诸证']"),
            ('阴血不足，血行不畅，腿脚挛急或腹中疼痛。伤寒脉浮，自汗出，小便数，心烦微恶寒，脚挛急，足温者。', "['阴血不足', '血行不畅', '腿脚挛急', '腹中疼痛', '伤寒脉浮', '自汗出', '小便数', '心烦微恶寒', '脚挛急', '足温']"),
            ('心气不足，思虑太过，肾经虚损，真阳不固，溺有余沥，小便白浊，梦寐频泄。', "['心气不足', '思虑太过', '肾经虚损', '真阳不固', '溺有余沥', '小便白浊', '梦寐频泄']"),
            ('久病体弱或吐下后胃虚有热，气逆不降，呃逆或呕吐，舌红脉虚数。', "['久病体弱', '吐下后胃虚有热', '气逆不降', '呃逆', '呕吐', '舌红', '脉虚数']"),
            ('里水，一身面目黄肿，其脉沉，小便不利。', "['里水', '一身面目黄肿', '脉沉', '小便不利']"),
            ('半产，恶寒战栗如灌水，虽蒙重被，尚鼓颔不止，须臾反烦热如灼，虽寒天欲得凉风；或腰腹疼痛，乍来乍止，其来也，如刺如割，如绞如啮，而流汗如雨，呻吟不已；或又渴，好热汤，而阴门下瘀液臭汁。', "['半产', '恶寒战栗', '灌水', '重被', '鼓颔不止', '反烦热如灼', '寒天欲得凉风', '腰腹疼痛', '刺割绞啮', '流汗如雨', '呻吟不已', '渴', '热汤', '阴门下瘀液臭汁']"),
            ('辰戌之岁，病身热头痛，呕吐，气郁中满，瞀闷少气，足痿，注下赤白，肌腠疮疡，发为痈疽。', "['辰戌之岁', '病身热头痛', '呕吐', '气郁中满', '瞀闷少气', '足痿', '注下赤白', '肌腠疮疡', '发为痈疽']")
        ]
        history += pre_history
        print('history = ', history, '\n')
        for data in tqdm(input_data2, desc="Processing"):
            symptoms_input_preprocess = data['symptoms_input_preprocess']
            input_answer = symptoms_input_preprocess
            #print('input = ', input_answer, '\n')
            symptoms_ouput, h= model.chat(generate_kwargs, input_answer, history=history)
            #print("Response: ", symptoms_ouput, '\n')
            data['symptoms_ouput'] = symptoms_ouput
            #print('*********************************\n')
            result.append(data)
            # 每完成多少个数据就更新到文件中
            if len(result) == 10:
                add_result_to_file(save_file_path_input_data2, result)
        add_result_to_file(save_file_path_input_data2, result)
        print('end!')

    # 处理inpu_data3, 无数字开头的数据，且只有二个句号（可理解为前面为证型，后面为证型描述）
    if mode == 3:
        background = "我希望你能够以一名中医领域专家的身份帮我完成如下的任务。首先我会给你一段文本，然后你需要尽可能多的提取出这段文本中的证型实体词和症状实体词，构成证型实体词列表和症状实体词列表。你的输出需要严格按照python列表的格式输出，我接下来会给你几个例子。你需要结合下列例子理解我的上述要求，并且按照要求完成任务。"
        example1 = "输入：脾胃气虚证。面色萎黄，语声低微，气短乏力，食少便溏，舌淡苔白，脉虚弱。输出：[['脾胃气虚证'], ['面色萎黄', '语声低微', '气短乏力', '食少便溏', '舌淡苔白', '脉虚弱']]"
        #example2 = "输入：血虚寒厥证。手足厥寒，或腰、股、腿、足、肩臂疼痛，口不渴，舌淡苔白，脉沉细或细而欲绝。输出：[['血虚寒厥证'], ['手足厥寒', '腰疼痛', '股疼痛', '腿疼痛', '足疼痛', '肩臂疼痛', '口不渴', '脉沉细']]" 
        prompt_base = background + example1
        print('prompt_base: ', prompt_base, '\n')
        symptoms_ouput, history= model.chat(generate_kwargs, prompt_base, history=history)
        print('prompt_output: ', symptoms_ouput, '\n')
        
        
        while True:
            query = input(">>>> ")
            if query == "q":
                break

            response, history = model.chat(generate_kwargs, query, history=history)
            print(f"Bot: {response}")
        # >>>> 你的输出格式不正确，正确的输出格式为：[['脾胃气虚证'], ['面色萎黄', '语声低微', '气短乏力', '食少便溏', '舌淡苔白', '脉虚弱']]
        print("进入识别...")

        pre_history = [
            ('风温初起，表热轻证。证见但咳，身热不甚，口微渴，苔薄白，脉浮数。', "[['风温初起', '表热轻证'], ['但咳', '身热不甚', '口微渴', '苔薄白', '脉浮数']]"),
            ('阴虚火旺盗汗。发热盗汗，面赤心烦，口干唇燥，大便干结，小便黄赤，舌红苔黄，脉数。', "[['阴虚火旺盗汗'], ['发热盗汗', '面赤心烦', '口干唇燥', '大便干结', '小便黄赤', '舌红苔黄', '脉数']]"),
            ('肝郁血虚脾弱证。两胁作痛，头痛目眩，口燥咽干，神疲食少，或月经不调，乳房胀痛，脉弦而虚者。', "[['肝郁血虚脾弱证'], ['两胁作痛', '头痛目眩', '口燥咽干', '神疲食少', '或月经不调', '乳房胀痛', '脉弦而虚']]"),
            ('肺热喘咳。气喘咳嗽，皮肤蒸热，日晡尤甚，舌红苔黄，脉细数。', "[['肺热喘咳'], ['气喘咳嗽', '皮肤蒸热', '日晡尤甚', '舌红苔黄', '脉细数']]"),
            ('脏躁。症见精神恍惚，常悲伤欲哭，不能自主，心中烦乱，睡眠不安，甚则言行失常，呵欠频作，舌淡红苔少，脉细微数。', "[['脏躁'], ['精神恍惚', '常悲伤欲哭', '不能自主', '心中烦乱', '睡眠不安', '言行失常', '呵欠频作', '舌淡红苔少', '脉细微数']]"),
            ('痰壅气逆食滞证。咳嗽喘逆，痰多胸痞，食少难消，舌苔白腻，脉滑。', "[['痰壅气逆食滞证'], ['咳嗽喘逆', '痰多胸痞', '食少难消', '舌苔白腻', '脉滑']]"),
            ('湿温初起及暑温夹湿之湿重于热证。头痛恶寒，身重疼痛，肢体倦怠，面色淡黄，胸闷不饥，午后身热，苔白不渴，脉弦细而濡。', "[['湿温初起及暑温夹湿之湿重于热证'], ['头痛', '身重疼痛', '肢体倦怠', '面色淡黄', '胸闷不饥', '午后身热', '苔白不渴', '脉弦细而濡']]"),
            ('肝气郁结证。症见胸膈胀闷，上气喘急，心下痞满，不思饮食，脉弦。', "[['肝气郁结证'], ['胸膈胀闷', '上气喘急', '心下痞满', '不思饮食', '脉弦']]"),
            ('风湿在表之痹证。肩背痛不可回顾，头痛身重，或腰脊疼痛，难以转侧，苔白，脉浮。', "[['风湿在表之痹证'], ['肩背痛不可回顾', '头痛身重', '腰脊疼痛', '难以转侧', '苔白', '脉浮']]"),
            ('风痰上扰证。眩晕，头痛，胸膈痞闷，恶心呕吐，舌苔白腻，脉弦滑。', "[['风痰上扰证'], ['眩晕', '头痛', '胸膈痞闷', '恶心呕吐', '舌苔白腻', '脉弦滑']]"),
            ('外感风寒表实证。恶寒发热，头身疼痛，无汗而喘，舌苔薄白，脉浮紧。', "[['外感风寒表实证'], ['恶寒发热', '头身疼痛', '无汗而喘', '舌苔薄白', '脉浮紧']]"),
            ('外感风寒湿邪，内有蕴热证。恶寒发热，无汗，头痛项强，肢体酸楚疼痛，口苦微渴，舌苔白或微黄，脉浮。', "[['外感风寒湿邪，内有蕴热证'], ['恶寒发热', '无汗', '头痛项强', '肢体酸楚疼痛', '口苦微渴', '舌苔白或微黄', '脉浮']]"),
            ('痹证日久，肝肾两虚，气血不足证。腰膝疼痛、痿软，肢节屈伸不利，或麻木不仁，畏寒喜温，心悸气短，舌淡苔白，脉细弱。', "[['痹证日久，肝肾两虚，气血不足证'], ['腰膝疼痛', '痿软', '肢节屈伸不利', '或麻木不仁', '畏寒喜温', '心悸气短', '舌淡苔白', '脉细弱']]"),
            ('肝血不足，虚热内扰证。虚烦失眠，心悸不安，头目眩晕，咽干口燥，舌红，脉弦细。', "[['肝血不足，虚热内扰证'], ['虚烦失眠', '心悸不安', '头目眩晕', '咽干口燥', '舌红', '脉弦细']]"),
            ('血虚寒凝，瘀血阻滞证。产后恶露不行，小腹冷痛。', "[['血虚寒凝，瘀血阻滞证'], ['产后恶露不行', '小腹冷痛']]"),
            ('心经火热证。心胸烦热，口渴面赤，意欲饮冷，以及口舌生疮；或心热移于小肠，小便赤涩刺痛，舌红，脉数。', "[['心经火热证'], ['心胸烦热', '口渴面赤', '意欲饮冷', '口舌生疮', '或', '心热移于小肠', '小便赤涩刺痛', '舌红', '脉数']]"),
            ('湿热黄疸。一身面目俱黄，黄色鲜明，发热，无汗或但头汗出，口渴欲饮，恶心呕吐，腹微满，小便短赤，大便不爽或秘结，舌红苔黄腻，脉沉数或滑数有力。', "[['湿热黄疸'], ['一身面目俱黄', '黄色鲜明', '发热', '无汗或但头汗出', '口渴欲饮', '恶心呕吐', '腹微满', '小便短赤', '大便不爽或秘结', '舌红苔黄腻', '脉沉数或滑数有力']]"),
            ('肝胃虚寒，浊阴上逆证。食后泛泛欲吐，或呕吐酸水，或干呕，或吐清涎冷沫，胸满脘痛，巅顶头痛，畏寒肢冷，甚则伴手足逆冷，大便泄泻，烦躁不宁，舌淡苔白滑，脉沉弦或迟。', "[['肝胃虚寒，浊阴上逆证'], ['食后泛泛欲吐', '呕吐酸水', '干呕', '吐清涎冷沫', '胸满脘痛', '巅顶头痛', '畏寒肢冷', '手足逆冷', '大便泄泻', '烦躁不宁', '舌淡苔白滑', '脉沉弦或迟']]"),
            ('气血两虚证。面色苍白或萎黄，头晕耳眩，四肢倦怠，气短懒言，心悸怔忡，饮食减少，舌淡苔薄白，脉细弱或虚大无力。', "[['气血两虚证'], ['面色苍白或萎黄', '头晕耳眩', '四肢倦怠', '气短懒言', '心悸怔忡', '饮食减少', '舌淡苔薄白', '脉细弱或虚大无力']]"),
            ('湿痰证。咳嗽痰多，色白易咯，恶心呕吐，胸膈痞闷，肢体困重，或头眩心悸，舌苔白滑或腻，脉滑。', "[['湿痰证'], ['咳嗽痰多', '色白易咯', '恶心呕吐', '胸膈痞闷', '肢体困重', '头眩心悸', '舌苔白滑或腻', '脉滑']]"),
        ]
        history += pre_history
        print('history = ', history)
        for data in tqdm(input_data3, desc="Processing"):
            symptoms_input_preprocess = data['symptoms_input_preprocess']
            input_answer = symptoms_input_preprocess
            #print('input = ', input_answer, '\n')
            symptoms_ouput, h= model.chat(generate_kwargs, input_answer, history=history)
            data['symptoms_ouput'] = symptoms_ouput
            result.append(data)
            # 每完成多少个数据就更新到文件中
            if len(result) == 10:
                add_result_to_file(save_file_path_input_data3, result)
        add_result_to_file(save_file_path_input_data3, result)
        print('end!')

    # 处理 input_data4, 无数字开头的数据，且有三个及以上的句号（数据不规范，后续处理, 以后77个，不行我给删掉）
    if mode == 4:
        # 有三个以上的直接删除吧
        for i, data in enumerate(input_data4):
            symptoms_input_preprocess = data['symptoms_input_preprocess']
            data['symptoms_ouput'] = ""
            result.append(data)
        add_result_to_file(save_file_path_input_data4, result)
        print('end!')

if __name__ == '__main__':
    main()


