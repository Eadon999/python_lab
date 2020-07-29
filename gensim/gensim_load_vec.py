import gensim.models.keyedvectors as word2vec1
import gensim
import json
import time
import numpy as np


def load_pretrained_vec(pathToBinVectors, limit, binary):
    s_t = time.time()
    embed_map = gensim.models.KeyedVectors.load_word2vec_format(pathToBinVectors, limit=limit, binary=binary)
    print("load vector file use:{}".format((time.time() - s_t) / 60))
    return embed_map


def read_feature_word(feature):
    with open(feature, encoding='utf-8') as f:
        feature = json.loads(f.readlines()[0])
    print("feature word num:{}".format(len(feature)))
    return feature


def find_from_pretrained_vec(fetaure_word):
    k = 0
    find_vec = []
    for i in fetaure_word:
        try:
            vec = embed_map.get_vector(i)
            res = "{} {}\n".format(i, " ".join([str(i) for i in vec.tolist()]))
            find_vec.append(res)

        except:
            k += 1
            print(i)
    print("not found num:", k)
    find_vec.insert(0, "{} {}\n".format(len(find_vec), 300))
    return find_vec


def write_to_vec(output, find_vec):
    with open(output, 'w', encoding='utf-8') as f:
        f.writelines(find_vec)


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


if __name__ == '__main__':
    feature_w_p = "./D:\PersonalGitProject\python_lab/new_feature_word.txt"
    output_p = ""
    pathToBinVectors = 'D:\wordVecFile\cc.zh.300.vec\cc.zh.300.vec'
    pathToBinVectors = 'D:\PersonalGitProject\python_lab\interest_entity20191128.w2v'
    limit = 200000
    embed_map = load_pretrained_vec(pathToBinVectors, None, True)
    # word_base = embed_map.get_vector("睡")
    # print(word_base)
    # word_compare = embed_map.get_vector("哄")
    # word_sleep = embed_map.get_vector("催眠")
    # word_getup = embed_map.get_vector("起床")
    # word_think = embed_map.get_vector("冥想")
    # print(word_base)
    # print(word_compare)
    # print(word_base + word_compare)
    # combine_vec = (word_base + word_compare) / 2
    # print(combine_vec)
    # print(word_compare)
    # embed_map.distance(word_base, word_compare)

    # feature_word = read_feature_word(feature_w_p)
    # finded_vec = find_from_pretrained_vec(feature_word)
    # write_to_vec(output_p, finded_vec)
    # print(cos_sim(word_getup, combine_vec))
    # print(cos_sim(word_think, combine_vec))
    # print(cos_sim(word_sleep, combine_vec))
    # print(cos_sim(word_sleep, word_getup))
    # print(embed_map.distance("催眠", "睡"))
    # print(cos_sim(word_sleep, word_base))
