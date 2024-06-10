from collections import Counter
import numpy as np
import pandas as pd
from nmt import get_data


def get_stats(ids):
        counter = Counter(zip(ids, ids[1:]))
        return counter
    
def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and pair[0] == ids[i] and pair[1] == ids[i + 1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def save_merges(vocab_size, min_length):
    num_merges = vocab_size - 256
    idx = 256
    ids = [int(byte) for byte in sentences_en.encode('utf-8')]
    print(ids)
    merges = np.zeros(shape=(num_merges, 3), dtype=np.int64)
    for i in range(num_merges):
        if min_length >= len(ids):
            break
        stats = get_stats(ids)
        pair = stats.most_common(1)[0][0]
        # print(pair)
        ids = merge(ids, pair, idx)
        # print(ids)
        merges[i, :] = pair[0], pair[1], idx
        idx += 1
        print(i)
    print(merges)

    merges_df = pd.DataFrame(merges, columns=['pair0', 'pair1', 'idx'])
    merges_df.to_csv('merges_en.csv', index=False)

sentences_en, sentences_es = get_data()
# sentences_es = [f'startofseq {s} endofseq' for s in sentences_es]
sentences_en = " ".join(sentences_en)
print(sentences_en)
save_merges(10000, 10)
