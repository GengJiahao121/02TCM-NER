from utils import Huozi
import json
import os
import re
import ast
from tqdm import tqdm

precision = "fp16"
huozi1_model_name_or_path = "/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/LLM/huozi-7b-sft"
huozi2_model_name_or_path = "/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/LLM/huozi-7b-rlhf"
instruct_dir = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/herb_infer.json'
infer_file_path = '/root/autodl-tmp/Huatuo-Llama-Med-Chinese-main/data/ouput_data/加减化裁_infer_output2.json'

model = Huozi(huozi2_model_name_or_path, precision)
generate_kwargs = {
    "max_new_tokens": 512,
    "temperature": 0.001,
    "do_sample": True,
    "repetition_penalty":  1.03,
    "top_k": 40,
    "top_p": 0.001,
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
    
    background = "我希望你能够以一名中医领域专家的身份帮我完成如下的任务。首先我会给你一段文本，你需要尽可能多的提取出这段文本中的症状实体词和中草药实体词，构成症状词实体列表和中草药实体列表。你的输出需要严格按照python列表的格式输出，分别为症状词实体词列表和中草药实体词列表。我接下来给你一个例子，你需要结合下列例子理解我的上述要求，并且按照要求完成任务。"
    example1 = "输入：若畏寒肢冷，脘腹疼痛者，加干姜、附子以温中祛寒。输出：症状词实体列表：['畏寒肢冷', '脘腹疼痛'] 中草药实体列表：['干姜', '附子']"
    example2 = "输入：若畏寒肢冷，脘腹疼痛者，加干姜、附子以温中祛寒。输出：[['畏寒肢冷', '脘腹疼痛'],['干姜', '附子']]" 

    prompt_base = background + example2
    print('prompt_base: ', prompt_base, '\n')
    addorsub_ouput, history= model.chat(generate_kwargs, prompt_base, history=history)
    print('prompt_base_output: ', addorsub_ouput, '\n')
    
    while True:
        query = input(">>>> ")
        if query == "q":
            break

        response, history = model.chat(generate_kwargs, query, history=history)
        print(f"Bot: {response}")

    print("进入加减化裁识别...")
    # >>> 你的输出格式不准确，准确的输出格式为：[['畏寒肢冷', '脘腹疼痛'], ['干姜', '附子']]

    pre_history = [
        ('胸膈痞满者，加枳壳、陈皮以行气宽胸', "[['胸膈痞满'], ['枳壳', '陈皮']]"),
        ('心悸失眠者，加酸枣仁以宁心安神', "[['心悸', '失眠'], ['酸枣仁']]"),
        ('若畏寒肢冷，脘腹疼痛者，加干姜、附子以温中祛寒', "[['畏寒肢冷', '脘腹疼痛'], ['干姜', '附子']]"),
        ('烦渴，加黄芪', "[['烦渴'], ['黄芪']]"),
        ('胃冷，呕吐涎味，加丁香', "[['胃冷', '呕吐涎味'], ['丁香']]"),
        ('呕逆，加藿香', "[['呕逆'], ['藿香']]"),
        ('若兼腹中痛者，加白芍以柔肝止痛', "[['兼腹中痛'], ['白芍']]"),
        ('头顶痛者，加藁本、细辛以疏风止痛', "[['头顶痛'], ['藁本', '细辛']]"),
        ('兼气滞者，加木香、枳壳以理气解郁', "[['兼气滞'], ['木香', '枳壳']]"),
        ('本方亦可用于虚人感冒，加苏叶少许以增辛散之力', "[['虚人感冒'], ['苏叶']]"),
        ('咳嗽者，加杏仁、橘红', "[['咳嗽'], ['杏仁', '橘红']]"),
        ('头痛风热甚者，去羌活，加蔓荆子、菊花', "[['头痛风热甚'], ['蔓荆子', '菊花']]"),
        ('头痛久而不愈者（邪深入络），加僵蚕、全蝎、桃仁、红花', "[['头痛久而不愈（邪深入络）'], ['僵蚕', '全蝎', '桃仁', '红花']]"),
        ('若肺、肾阴虚，加麦冬、地黄、玄参以金水并凋，滋阴清热', "[['肺', '肾阴虚'], ['麦冬', '地黄', '玄参']]"),
        ('若胞睑起泡较多，宜加利水渗湿之品，如茯苓、泽泻、车前子等', "[['胞睑起泡较多'], ['茯苓', '泽泻', '车前子']]"),
        ('胃部灼热，加蒲公英15g，灼热喜按喜温饮，加高良姜3g，胸部痞满发胀，加九香虫3g', "[['胃部灼热', '灼热喜按喜温饮', '胸部痞满发胀'], ['蒲公英', '高良姜', '九香虫']]"),
        ('吐酸水，加生牡蛎30g，或瓦楞子30g', "[['吐酸水'], ['生牡蛎', '瓦楞子']]"),
        ('若伴口黏大便不爽、头痛重着如裹者，可加藿香、苍术、肉豆蔻、佩兰等以加强健脾芳香化湿', "[['伴口黏', '大便不爽', '头痛重着如裹'], ['藿香', '苍术', '肉豆蔻', '佩兰']]"),
        ('黑瘦人，合四物汤，加大枫子、黄柏', "[['黑瘦人'], ['四物汤', '大枫子', '黄柏']]"),
        ('肥白人，加荆芥、防风、羌活、白芷、苍术，取其能胜湿也', "[['肥白人'], ['荆芥', '防风', '羌活', '白芷', '苍术']]"),
        ('如心下痞，每服加枳实一钱', "[['心下痞'], ['枳实']]"),
        ('心火盛，烦惊不寐，加竹叶、黄连。', "[['心火盛', '烦惊不寐'], ['竹叶', '黄连']]"),
        ('此症用宣扬散亦佳。柴胡一钱，荆芥二钱。当归一两，麦冬一两，天花粉三钱。', "[['此症用宣扬散亦佳'], ['柴胡', '荆芥', '当归', '麦冬', '天花粉']]"),
        ('如气郁化火而见胎热气喘、口苦咽干、尿黄、舌红苔黄者，加黄芩、合欢花、柴胡以疏肝清热', "[['气郁化火', '胎热气喘', '口苦咽干', '尿黄', '舌红苔黄'], ['黄芩', '合欢花', '柴胡']]"),
        ('若头脑痛甚者，加羌活八分，葱白二根', "[['头脑痛甚'], ['羌活', '葱白']]"),
        ('自汗恶风者，加桂枝、白芍各一钱', "[['自汗恶风'], ['桂枝', '白芍']]"),
        ('因本方涌吐之功较弱，若用于痰食壅滞胸脘之重证，可酌加瓜蒂、藜芦，以增强其涌吐作用。', "[['涌吐之功较弱', '痰食壅滞胸脘之重证'], ['瓜蒂', '藜芦']]"),
    ]
    history = history + pre_history
    #print('history: ', history, '\n')
    for d in tqdm(input_data, desc="Processing"):    
        if 'ADDorSUB_input' in d:
            addorsub_input = d['ADDorSUB_input']

            # 使用正则表达式拆分每种情况
            sentences = re.split(r'[。；]', addorsub_input)
             # 去除空白项
            sentences = [s.strip() for s in sentences if s.strip()]


            # 对每个情况进行实体识别（症状和对应的中草药结合一起实体识别）
            sentences_ouput = []
            for sentence in sentences:
                input_answer = sentence
                sentence_ouput, h= model.chat(generate_kwargs, input_answer, history=history)
                try:
                    sentence_ouput = ast.literal_eval(sentence_ouput)
                except (ValueError, SyntaxError) as e:
                    print(f"Error: {e}")
                    print('symptoms_ouput = ', sentence_ouput)
                    print('******')
                    
                sentences_ouput.append(sentence_ouput)

            d['ADDorSUB_output'] = sentences_ouput
            result.append(d)

            # 每完成多少个数据就更新到文件中
            if len(result) == 10:
                add_result_to_file(infer_file_path, result)
    add_result_to_file(infer_file_path, result)
    print('end!')


if __name__ == '__main__':
    main()



#response, history = model.chat(generate_kwargs, query, history=history)
#print(f"Bot: {response}")

'''
[
    {
        "index": 1,
        "方名": "四君子汤",
        "介绍": "四君子汤，中医方剂学。出自《太平惠民和剂局方》。为补益剂。具有益气健脾之功效。主治脾胃气虚证。症见面色萎黄，语声低微，气短乏力，食少便溏，舌淡苔白，脉虚数。临床常用于治疗慢性胃炎、消化性溃疡等属脾胃气虚者。",
        "歌诀": "四君子汤中和义，参术茯苓甘草比，益以夏陈名六君，祛痰补益气虚饵，除却半夏名异功，或加香砂气滞使。",
        "组成": "人参去芦，白术、茯苓去皮（各9g），甘草炙（6g）。",
        "herb_list": ["人参", "白术", "茯苓", "甘草"],
        "用法用量": "1、现代用法：水煎服。2、古代用法：上为细末。每服两钱，水一盏，煎至七分，通口服，不拘时候；入盐少许，白汤点亦得。",
        "功用": "益气健脾。",
        "主治": "脾胃气虚证。面色萎黄，语声低微，气短乏力，食少便溏，舌淡苔白，脉虚弱。",
        "syndrome_and_symptoms_list": [
            {"syndrome": ["脾胃气虚证"], "symptoms": ["面色萎黄", "语声低微", "气短乏力", "食少便溏", "舌淡苔白", "脉虚弱"]}
          ],
        "方义": "本证多由脾胃气虚，运化乏力所致，治疗以益气健脾为主。脾胃为后天之本，气血生化之源，脾胃气虚，受纳与健运乏力，则饮食减少；湿浊内生，脾胃运化不利，故大便溏薄；脾主肌肉，脾胃气虚，四肢肌肉无所禀受，故四肢乏力；气血生化不足，不能荣于面，故见面色萎白；脾为肺之母，脾胃一虚，肺气先绝，故见气短、语声低微；舌淡苔白，脉虚弱均为气虚之象。正如《医方考》所说：“夫面色萎白，则望之而知其气虚矣；言语轻微，则闻之而知其气虚矣；四肢无力，则问之而知其气虚矣；脉来虚弱，则切之而知其气虚矣。”方中人参为君，甘温益气，健脾养胃。臣以苦温之白术，健脾燥湿，加强益气助运之力；佐以甘淡茯苓，健脾渗湿，苓术相配，则健脾祛湿之功益著。使以炙甘草，益气和中，调和诸药。四药配伍，共奏益气健脾之功。",
        "配伍特点": "本方重在补益脾胃之虚，兼以苦燥淡渗以祛湿浊，颇合脾欲缓、喜燥恶湿之性。",
        "运用": "1、本方为治疗脾胃气虚证的基础方，后世众多补脾益气方剂多从此方衍化而来。临床应用以面白食少，气短乏力，舌淡苔白，脉虚弱为辨证要点。2、本方常用于慢性胃炎、胃及十二指肠溃疡等属脾气虚者。",
        "加减化裁": "若呕吐，加半夏以降逆止呕；胸膈痞满者，加枳壳、陈皮以行气宽胸；心悸失眠者，加酸枣仁以宁心安神；若畏寒肢冷，脘腹疼痛者，加干姜、附子以温中祛寒。烦渴，加黄芪；胃冷，呕吐涎味，加丁香；呕逆，加藿香；脾胃不和，倍加白术、姜、枣；脾困，加人参、木香、缩砂仁；脾弱腹胀，不思饮食，加扁豆、粟米；伤食，加炒神曲；胸满喘急，加白豆蔻。",
        "add_or_sub_list": [
            [["呕吐"], ["半夏"]],
            [["胸膈痞满"], ["枳壳", "陈皮"]],
            [["心悸失眠"], ["酸枣仁"]],
            [["畏寒肢冷", "脘腹疼痛"], ["干姜", "附子"]],
            [["烦渴"], ["黄芪"]],
            [["胃冷", "呕吐涎味"], ["丁香"]],
            [["呕逆"], ["藿香"]],
            [["脾胃不和"], ["白术", "姜", "枣"]],
            [["脾困"], ["人参", "木香", "缩砂仁"]],
            [["脾弱腹胀", "不思饮食"], ["扁豆", "粟米"]],
            [["伤食"], ["炒神曲"]],
            [["胸满喘急"], ["白豆蔻"]]
        ],
        "化裁方之间的鉴别": "异功散、六君子汤、香砂六君子汤均由四君子汤加味而成。均有益气健脾之功。异功散中加陈皮，兼行气化滞，适用于脾胃气虚兼气滞证；六君子汤中加陈皮、半夏，兼燥湿和胃，适用于脾胃气虚兼痰湿证；香砂六君子汤中加陈皮、半夏、木香、砂仁，功在益气和胃，行气化痰，适用于脾胃气虚、痰阻气滞证。保元汤，以补气药为主，配伍少量肉桂以助阳，功能益气温阳，适用于小儿元气不足之证。",
        "重要文献摘要": "1、原书主治《太平惠民和剂局方》卷3：“荣卫气虚，脏腑怯弱。心腹胀满，全不思食，肠鸣泄泻，呕哕吐逆，大宜服之。”2、方论选录汪昂《医方集解·补养之剂》：“此手足太阴、足阳明药也。人参甘温，大补元气为君。白术苦温，燥脾补气为臣。茯苓甘淡，渗湿泻热为佐。甘草甘平，和中益土为使也。气足脾运，饮食倍进，则余脏受荫，而色泽身强矣。再加陈皮以理气散逆，半夏以燥湿除痰，名日六君，以其皆中和之品，故日君子也。”"
    },
    ... ... 
]
'''