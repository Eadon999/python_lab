import gensim.models.keyedvectors as word2vec1
import gensim
import json
import time
import numpy as np
import pandas as pd


def load_pretrained_vec(pathToBinVectors, limit, binary):
    s_t = time.time()
    embed_map = gensim.models.KeyedVectors.load_word2vec_format(pathToBinVectors, limit=limit, binary=binary)
    print("load vector file use:{}".format((time.time() - s_t) / 60))
    return embed_map


def read_csv(path):
    data_df = pd.read_csv(path, delimiter='\t')
    return data_df


if __name__ == '__main__':
    play_title_path = r'D:\工作资料\20200706数据探索\rt_play_top_detail.csv'
    pathToBinVectors = r'D:\工作资料\20200706数据探索\audiotitlevec.w2v'
    limit = 200000
    # embed_map = load_pretrained_vec(pathToBinVectors, 1000, True)
    data = read_csv(play_title_path)
    print(data.shape)
    user_ids = []
    not_in_vob = []
    for idx, row in data.iterrows():
        play_info = json.loads(row['rt_play_top'])
        user_info = row['device_id']
        for item_dict in play_info:
            audio_id = str(item_dict.get('audio_id'))
            try:
                sim_res = embed_map.similar_by_word(audio_id, topn=100)
                if len(sim_res) >= 10:
                    user_ids.append(user_info)
            except:
                not_in_vob.append(audio_id + '\n')

    print("有召回结果的用户数:{}, 有画像的用户数:{}".format(len(set(user_ids)), len(set(data['device_id']))))
    with open('./out_of_audio.txt', 'w') as f:
        for ov in not_in_vob:
            f.write(ov)
