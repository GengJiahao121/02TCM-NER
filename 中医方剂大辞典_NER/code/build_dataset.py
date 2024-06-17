
import json
result = []
index = 0
with open("/root/autodl-tmp/huozi/中医方剂大辞典/data/prescriptions.txt", 'r') as f:
    for line in f:
        line = line.strip()
        prescription = line.split('\t')
        sample = {}
        if len(prescription) == 2:
            index += 1
            symptoms = prescription[0]
            herbs = prescription[1]
            sample['id'] = index
            sample['symptoms_sequence'] = symptoms.strip()
            sample['herbs_sequence'] = herbs.strip()
            result.append(sample)

with open("/root/autodl-tmp/huozi/中医方剂大辞典/data/prescriptions.json", 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)


        
        