
import json
import os
from tqdm import tqdm
import ast

import sys
sys.path.append('/root/autodl-tmp/huozi')
from utils import Huozi

precision = "fp16" # 训练精度
huozi1_model_name_or_path = "/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/LLM/huozi-7b-sft" # 模型训练参数路径
huozi2_model_name_or_path = "/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/LLM/huozi-7b-rlhf" 
infer_file_path = '/root/autodl-tmp/huozi/中医方剂大辞典/data/new_prescriptions.json' # 输入文件路径

# 模型加载
model = Huozi(huozi2_model_name_or_path, precision)
history = None
generate_kwargs = {
    "max_new_tokens": 512,
    "temperature": 0.001,
    "do_sample": True,
    "repetition_penalty":  1.03,
    "top_k": 40,
    "top_p": 0.01,
}

# 将结果写入文件
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

instruct_dir = '/root/autodl-tmp/huozi/中医方剂大辞典/data/prescriptions.json'
def main():
    with open(instruct_dir, 'r') as f:
        input_data = json.load(f)

    result = []
    history = None

    background = "我希望你能够以一名中医领域专家的身份帮我完成如下的任务。首先我会给你一段文本，然后你需要尽可能多的提取出这段文本中的中草药实体词，比如人参，枸杞，桂圆，黄芪等。你的输出需要严格按照python列表的格式输出，我接下来会给你几个例子。你需要结合下列例子理解我的上述要求，并且按照要求完成任务。"
    # example1 = "输入：当归三两（9g）、芍药一斤（48g）、茯苓四两（12g）、白术四两（12g）。 输出：['当归', '芍药', '茯苓', '白术']"
    example2 = "输入：半夏（洗）半升（9g），生姜（切）五两（15g），枳实（炙）四枚（9g），大枣（擘）十二枚（4枚），大黄二两（6g）。 输出：['半夏', '生姜', '枳实', '大枣', '大黄']" 
    prompt_base = background + example2
    print('prompt_base: ', prompt_base, '\n')
    herb_ouput, history= model.chat(generate_kwargs, prompt_base, history=history)
    print('prompt_base_output: ', herb_ouput, '\n')

    while True:
        query = input(">>>> ")
        if query == "q":
            break

        response, history = model.chat(generate_kwargs, query, history=history)
        print(f"Bot: {response}")

    print("进入加减化裁识别...")
    # >>> 你的输出格式不准确，准确的输出格式为：['半夏', '生姜', '枳实', '大枣', '大黄']

    pre_history = [
        ('决明子（炒）、柴胡（去苗）、黄连（去须）、苦竹叶、防风（去叉）、升麻各七钱五分，细辛（去苗）二钱五分，菊花、甘草（炙）各五钱。', "['决明子', '柴胡', '黄连', '苦竹叶', '防风', '升麻', '细辛', '菊花', '甘草']"),
        ('粟壳(醋炒)四两，杏仁二两，五味(焙干)一两，枯矾五钱。', "['粟壳', '杏仁', '五味', '枯矾]"),
        ('生地黄、川当归、白芍药各二钱，黄柏、知母各一钱，条芩、黄连、川芎、阿胶(炒)各八分，艾叶、香附、炙甘草各七分。', "['生地黄', '川当归', '白芍药', '黄柏', '知母', '条芩', '黄连', '川芎', '阿胶', '艾叶', '香附', '炙甘草']"),
        ('乱发一两。', "['乱发']"),
        ('生香附三钱（9g），木香二钱（6g），砂壳一钱半（4.5g），朴花、茅术须各二钱（各6g），五加皮、云苓皮各三钱（各9g），桑枝五钱（15g）。', "['生香附', '木香', '砂壳', '朴花', '茅术须', '五加皮', '云苓皮', '桑枝']"),
        ('黄连五分（1.5克），金银花二钱（6克），赤芍一钱（3克），丹皮二钱（6克），连翘一钱五分（4.5克），大贝二钱（6克），花粉二钱（6克），菊花二钱（6克），薄荷一钱（3克），甘草五分（1.5克），淡竹叶二十片。', "['黄连', '金银花', '赤芍', '丹皮', '连翘', '大贝', '花粉', '菊花', '薄荷', '甘草', '淡竹叶']"),
        ('黄芩(去黑心)、甘遂(麸炒黄)、龙胆(去芦头)各一两。', "['黄芩', '甘遂', '龙胆']"),
        ('红花、忍冬各7.5g，黄芩、连翘各0.6g，槟榔0.5g，木通、桔梗各0.3g，大黄0.9g。', "['红花', '忍冬', '黄芩', '连翘', '槟榔', '木通', '桔梗', '大黄']"),
        ('香附(分四份：一童便，一米醋，一人乳，一盐酒浸）五两（150g），蕲艾(醋煮）、当归各二两（60g），川芎、白芍、熟地黄(酒蒸)、黄芩各一两半（45g），阿胶(酒蒸)、臭椿根皮各一两（30g）。', "['香附', '蕲艾', '当归', '川芎', '白芍', '熟地黄', '黄芩', '阿胶', '臭椿根皮']"),
        ('柴胡、羌活、桔梗、金银花、连翘、防风、荆芥、薄荷叶、川芎、独活、前胡、白茯苓、甘草、枳壳。', "['柴胡', '羌活', '桔梗', '金银花', '连翘', '防风', '荆芥', '薄荷叶', '川芎', '独活', '前胡', '白茯苓', '甘草', '枳壳']"),
        ('羌活、防风、桑螵蛸、栀子、薄荷、当归、赤芍药、甘草、麻黄、连翘、菊花、木贼、白蒺藜、川芎、大黄、黄芩、荆芥各一两。', "['羌活', '防风', '桑螵蛸', '栀子', '薄荷', '当归', '赤芍药', '甘草', '麻黄', '连翘', '菊花', '木贼', '白蒺藜', '川芎', '大黄', '黄芩', '荆芥']"),
        ('白术、人参、黄芪、山药、茯苓各二钱（6g），紫河车三钱（9g），当归、丹皮、枣仁、远志各一钱五分（4.5g）。', "['白术', '人参', '黄芪', '山药', '茯苓', '紫河车', '当归', '丹皮', '枣仁', '远志']"),
        ('辰锦朱砂、白茯苓、黄芩、山栀子仁、人参各一两，虎睛（用仁）一对，牛黄脑麝、犀角屑各一分，钩藤、大黄（用湿纸裹煨）各四两。', "['辰锦朱砂', '白茯苓', '黄芩', '山栀子仁', '人参', '虎睛', '牛黄脑麝', '犀角屑', '钩藤', '大黄']"),
        ('炒厚朴、醋炒青皮、陈皮、使君子、槟榔、醋炒三棱、炒甘草各五钱，炒神曲、炒香附、黄连、炒麦芽、土炒白术、醋炒蓬术各一两，山楂一两半。', "['炒厚朴', '醋炒青皮', '陈皮', '使君子', '槟榔', '醋炒三棱', '炒甘草', '炒神曲', '炒香附', '黄连', '炒麦芽', '土炒白术', '醋炒蓬术', '山楂']"),
        ('大附子(童便煮一炷香)，人参(去芦)三分，桔梗一钱，生地黄一钱，蛤粉五分，玄参七分，升麻四分。', "['大附子', '人参', '桔梗', '生地黄', '蛤粉', '玄参', '升麻']"),
        ('蒜(去皮，切，水四升，煮取一升，去滓)六斤四两，酥(纳蒜汁中)一升，牛乳二升，荜茇、胡椒、干姜各三两，石蜜、阿魏、戎盐各二两，石上菖蒲、木香各一两', "['蒜', '酥', '牛乳', '荜茇', '胡椒', '干姜', '石蜜', '阿魏', '戎盐', '石上菖蒲', '木香']"),
        ('乌药、灵脂、当归、熟地、白芍、川芎、三棱、香附、甘草、元胡、陈皮、官桂、厚朴、防风。', "['乌药', '灵脂', '当归', '熟地', '白芍', '川芎', '三棱', '香附', '甘草', '元胡', '陈皮', '官桂', '厚朴', '防风']"),
        ('茯苓、泽泻、木通各二钱，猪苓、栀子(或倍之)、枳壳、车前子各一钱。', "['茯苓', '泽泻', '木通', '猪苓', '栀子', '枳壳', '车前子']"),
        ('阳起石(煅)，细辛(去叶)，赤石脂(煅)，川椒(去目合口，炒出汗)，肉豆蔻(面裹，煨)，白矾(枯)，干姜(炮，洗)，附子(炮，去皮脐)半两，硫黄(别研)三两。', "['阳起石', '细辛', '赤石脂', '川椒', '肉豆蔻', '白矾', '干姜', '附子', '硫黄']"),
        ('石斛(去根)、杜仲(去粗皮，微炙，锉)、牛膝(去苗)各一两半，远志(去心)、覆盆子、防风(去芦头)、薯蓣、五味子、山茱萸各三分，泽泻、白龙骨、萆薢(锉)、石龙芮、黄芪(锉)、附子(炮裂，去皮脐)、补骨脂(微炒)、人参(去芦头)、车前子、桂心、白茯苓、熟干地黄、肉苁蓉(酒浸一宿，刮去皱皮，炙干)、巴戟、蛇床子各一两，磁石(烧，醋淬七遍，捣碎，水飞过)、钟乳粉、鹿茸、菟丝子(酒浸三宿，晒干别捣为末)各二两，甘草(炙微赤，锉)半两。', "['石斛', '杜仲', '牛膝', '远志', '覆盆子', '防风', '薯蓣', '五味子', '山茱萸', '泽泻', '白龙骨', '萆薢', '石龙芮', '黄芪', '附子', '补骨脂', '人参', '车前子', '桂心', '白茯苓', '熟干地黄', '肉苁蓉', '巴戟', '蛇床子', '磁石', '钟乳粉', '鹿茸', '菟丝子', '甘草']")
    ]

    history = history + pre_history
    print('history = ', history, '\n') 

    for d in tqdm(input_data, desc='Prpcessing...'):
        if d['id'] <= 44062:
            continue
        herb_input = d['herbs_sequence']
        input_answer = herb_input
        herb_output, h= model.chat(generate_kwargs, input_answer, history=history)
        try:
            herb_output = ast.literal_eval(herb_output)
            d['herb_list'] = herb_output
            result.append(d)
        except (ValueError, SyntaxError) as e:
            print(f"Error: {e}")
            print('herb_output = ', herb_output)
            print('******')
        #print("Response: ", herb_ouput, '\n')
        # 每完成多少个数据就更新到文件中
        if len(result) == 10:
            add_result_to_file(infer_file_path, result)
    add_result_to_file(infer_file_path, result)
    print('end!')

if __name__ == '__main__':
    main()

