import os
import json
import collections

work_dir = os.getcwd()
data_dir = os.path.abspath("./data/crosswoz")

train_path = os.path.join(data_dir, "train.json")
val_path = os.path.join(data_dir, "val.json")
test_path = os.path.join(data_dir, "test.json")

train_nlu_path = os.path.join(data_dir, "nlu/train_nlu.json")
val_nlu_path = os.path.join(data_dir, "nlu/val_nlu.json")
test_nlu_path = os.path.join(data_dir, "nlu/test_nlu.json")

if __name__ == "__main__":
    all_intents = collections.defaultdict(int)
    all_tags = collections.defaultdict(int)
    for path, save_path in zip([train_path, val_path, test_path], [train_nlu_path, val_nlu_path, test_nlu_path]):
        with open(path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
            new_data = {}
            for sess_id, sess in data.items():
                new_sess = []
                for i, turn in enumerate(sess['messages']):
                    utterance = turn['content']
                    intents = []
                    tags = ['O'] * len(utterance)
                    for intent, domain, slot, slot_value in turn['dialog_act']:
                        intents.append(intent)
                        all_intents[intent] += 1
                        if slot_value in utterance:
                            start_idx = utterance.index(slot_value)
                            tags[start_idx] = 'B-'+ slot
                            if len(slot_value) != 1:
                                tags[start_idx + 1: start_idx + len(slot_value)] = ['I-'+ slot] * (len(slot_value)-1)
                    for t in tags:
                        all_tags[t] += 1
                    new_sess.append(
                        {
                            "uttr_id": i,
                            'role': turn['role'],
                            "utterance": utterance,
                            "intents":intents,
                            "tags": " ".join(tags),
                            "action":turn['dialog_act'],
                        }
                    )
                new_data[sess_id] = new_sess
            print(f"data size: {len(new_data)}")
            with open(save_path, 'w', encoding='utf-8') as fout:
                json.dump(new_data, fout, indent=2, ensure_ascii=False)
    print("data done")
    all_intents = sorted(all_intents.items(), key=lambda x:x[1], reverse=True)
    all_tags = sorted(all_tags.items(), key=lambda x:x[1], reverse=True)
    all_tags = ["PAD", "UNK"] + [t[0] for t in all_tags]
    with open(os.path.join(data_dir, "nlu/intents_vocab.txt"), "w", encoding="utf-8") as f:
        for b in all_intents:
            f.write(b[0] + "\n")
    with open(os.path.join(data_dir, "nlu/slots_vocab.txt"), "w", encoding="utf-8") as f:
        for b in all_tags:
            f.write(b + "\n")
    print("done")
        
 


