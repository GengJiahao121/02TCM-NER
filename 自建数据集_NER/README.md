## 自建数据集的构建过程

    注意：

    %%%

    **下面的文件路径有的不对，自己调整**

    %%%

主要涉及到对"主治"、"组成"、"加减化裁"中的的证型、症状、中草药实体词的提取。

前期先对各种情况进行分割、再进行实体识别。

### 对原始数据集进行数据分割

原始数据集

`/root/autodl-tmp/huozi_to_TCM-NER/自建数据集_NER/data/src_data/prescriptions.json`

### 主治

1、从原始数据集中提取 `主治`数据 得到 `/root/autodl-tmp/huozi_to_TCM-NER/自建数据集_NER/data/symptoms_infer.json`

转换代码：`/root/autodl-tmp/huozi_to_TCM-NER/自建数据集_NER/data/src_data/extract_some_attributes_from_pres.py`

2、对 `主治` 数据进行数据分割 并 实体识别

1. 数据分割 + 实体识别

`/root/autodl-tmp/huozi_to_TCM-NER/自建数据集_NER/code/process_symptom.py`

得到针对每种情况单独进行实体识别的数据集(详见代码)

### 组成

这个的处理相对较容易，因为只有一种情况

1、从原始数据集中提起 `组成` 数据 得到 `/root/autodl-tmp/huozi_to_TCM-NER/自建数据集_NER/data/herb_input.json`

提取代码：`/root/autodl-tmp/huozi_to_TCM-NER/自建数据集_NER/data/src_data/herb_to_infer.py`

2、对 `组成` 数据进行实体识别

详见代码：`/root/autodl-tmp/huozi_to_TCM-NER/自建数据集_NER/code/process_herb.py`

### 加减化裁

这个的输入数据也为`/root/autodl-tmp/huozi_to_TCM-NER/自建数据集_NER/data/herb_input.json`，用其中的`加减化裁`

1、 对 `加减化裁` 数据进行实体识别（症状和对应的中草药结合一起实体识别）

详见代码：`/root/autodl-tmp/huozi_to_TCM-NER/自建数据集_NER/code/process_加减化裁.py`

### 对 主治、组成、加减化裁 分别处理的输出结果合并成一个完整的数据集

`/root/autodl-tmp/huozi_to_TCM-NER/自建数据集_NER/data/ouput_data/merge.py`

得到最终的数据集result3.json

### 对result.json 中 主治、组成、加减化裁 处理的结果进行数据的清洗

详见代码：

`/root/autodl-tmp/huozi_to_TCM-NER/自建数据集_NER/data/ouput_data/clean.py`

得到最终的`cleaned_result.json`













