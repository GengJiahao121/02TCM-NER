## 通过“活字”大模型对中医方剂大辞典原始数据集进行实体识别过程

1、 中医方剂大辞典原始数据集

`./data/prescriptions.txt`

格式：

`症状   中草药集合`

例：

`豆疮黑陷，或变紫暗色，证在急危者。[TAB/空格]穿山甲（汤浸透，取甲锉碎，同热灰铛内慢火炒令黄色）五钱  红色曲（炒）  川乌（一枚，灰火中带焦炮）各二钱半
`

2、通过build_dataset.py进行数据处理得到perscriptions.jcon

`./code/build_dataset.py` -> `./data/perscriptions.jcon`

-> 

```
 {
        "id": 1,
        "symptoms_sequence": "豆疮黑陷，或变紫暗色，证在急危者。",
        "herbs_sequence": "穿山甲（汤浸透，取甲锉碎，同热灰铛内慢火炒令黄色）五钱  红色曲（炒）  川乌（一枚，灰火中带焦炮）各二钱半"
 },
 ... ...
```

3、通过复现“活字”大模型 对 prescriptions.json中的 herbs_sequence 进行中草药实体识别 结果保存到new_prescriptions.json

复现及实体识别代码：

`./code/process_herb.json`

结果：

`./data/new_prescriptions.json`

```
[
  {
    "id": 1,
    "symptoms_sequence": "豆疮黑陷，或变紫暗色，证在急危者。",
    "herbs_sequence": "穿山甲（汤浸透，取甲锉碎，同热灰铛内慢火炒令黄色）五钱  红色曲（炒）  川乌（一枚，灰火中带焦炮）各二钱半",
    "herb_list": [
      "穿山甲",
      "红色曲",
      "川乌"
    ]
  },
  ... ...
```

