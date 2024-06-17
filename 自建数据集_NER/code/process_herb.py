from utils import Huozi
import json
import os
from tqdm import tqdm
import ast

'''
注意：这个代码是后来我修正过的代码，之前没有pre_history

如果以后重新跑，pre_history中的内容要补全！！其他的应该没什么问题

'''

precision = "fp16"
huozi1_model_name_or_path = "/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/LLM/huozi-7b-sft"
huozi2_model_name_or_path = "/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/LLM/huozi-7b-rlhf"
infer_file_path = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/herb_infer_output1.json'
instruct_dir = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/herb_input.json'

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
    with open(instruct_dir, 'r') as f:
        input_data = json.load(f)

    result = []
    history = None

    background = "我希望你能够以一名中医领域专家的身份帮我完成如下的任务。首先我会给你一段文本，然后你需要尽可能多的提取出这段文本中的中草药实体词，比如人参，枸杞，桂圆，黄芪等。\
        你的输出需要严格按照python列表的格式输出，我接下来会给你几个例子。你需要结合下列例子理解我的上述要求，并且按照要求完成任务。"
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
        ('黄连五分（1.5克），金银花二钱（6克），赤芍一钱（3克），丹皮二钱（6克），连翘一钱五分（4.5克），大贝二钱（6克），花粉二钱（6克），菊花二钱（6克），薄荷一钱（3克），甘草五分（1.5克），淡竹叶二十片。', "['黄连', '金银花', \
         '赤芍', '丹皮', '连翘', '大贝', '花粉', '菊花', '薄荷', '甘草', '淡竹叶']"),
        ('黄芩(去黑心)、甘遂(麸炒黄)、龙胆(去芦头)各一两。', "['黄芩', '甘遂', '龙胆']"),
        ('红花、忍冬各7.5g，黄芩、连翘各0.6g，槟榔0.5g，木通、桔梗各0.3g，大黄0.9g。', "['红花', '忍冬', '黄芩', '连翘', '槟榔', '木通', '桔梗', '大黄']"),
        ('香附(分四份：一童便，一米醋，一人乳，一盐酒浸）五两（150g），蕲艾(醋煮）、当归各二两（60g），川芎、白芍、熟地黄(酒蒸)、黄芩各一两半（45g），阿胶(酒蒸)、臭椿根皮各一两（30g）。', "['香附', '蕲艾', '当归', '川芎', '白芍', \
         '熟地黄', '黄芩', '阿胶', '臭椿根皮']"),
        ('柴胡、羌活、桔梗、金银花、连翘、防风、荆芥、薄荷叶、川芎、独活、前胡、白茯苓、甘草、枳壳。', "['柴胡', '羌活', '桔梗', '金银花', '连翘', '防风', '荆芥', '薄荷叶', '川芎', '独活', '前胡', '白茯苓', '甘草', '枳壳']"),
        ('羌活、防风、桑螵蛸、栀子、薄荷、当归、赤芍药、甘草、麻黄、连翘、菊花、木贼、白蒺藜、川芎、大黄、黄芩、荆芥各一两。', "['羌活', '防风', '桑螵蛸', '栀子', '薄荷', '当归', '赤芍药', '甘草', '麻黄', '连翘', '菊花', '木贼', \
         '白蒺藜', '川芎', '大黄', '黄芩', '荆芥']"),
        ('白术、人参、黄芪、山药、茯苓各二钱（6g），紫河车三钱（9g），当归、丹皮、枣仁、远志各一钱五分（4.5g）。', "['白术', '人参', '黄芪', '山药', '茯苓', '紫河车', '当归', '丹皮', '枣仁', '远志']"),
        ('辰锦朱砂、白茯苓、黄芩、山栀子仁、人参各一两，虎睛（用仁）一对，牛黄脑麝、犀角屑各一分，钩藤、大黄（用湿纸裹煨）各四两。', "['辰锦朱砂', '白茯苓', '黄芩', '山栀子仁', '人参', '虎睛', '牛黄脑麝', '犀角屑', '钩藤', '大黄']"),
        ('炒厚朴、醋炒青皮、陈皮、使君子、槟榔、醋炒三棱、炒甘草各五钱，炒神曲、炒香附、黄连、炒麦芽、土炒白术、醋炒蓬术各一两，山楂一两半。', "['炒厚朴', '醋炒青皮', '陈皮', '使君子', '槟榔', '醋炒三棱', '炒甘草', '炒神曲', '炒香附', \
         '黄连', '炒麦芽', '土炒白术', '醋炒蓬术', '山楂']")
    ]

    history = history + pre_history
    print('history = ', history, '\n') 

    for d in tqdm(input_data, desc='Prpcessing...'):
        herb_input = d['herb_input']
        input_answer = herb_input
        herb_output, h= model.chat(generate_kwargs, input_answer, history=history)
        try:
            herb_output = ast.literal_eval(herb_output)
            d['herb_ouput'] = herb_output
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

