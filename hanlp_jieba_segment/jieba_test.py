import jieba
import jieba.posseg as pseg
import jieba.analyse
from pyhanlp import *

words1 = jieba.cut("红楼梦之迎春悲哀嫁错人", HMM=False)

words2 = jieba.cut("百年孤独1")
words4 = pseg.cut("百年孤独1", HMM=False)
words3 = jieba.cut("萧忆情-死性不改")
words5 = pseg.lcut("萧忆情-死性不改", HMM=False)

sentence = """红楼梦之迎春悲哀嫁错人"""

keywords = jieba.analyse.extract_tags(sentence, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns'))
keywords2 = jieba.analyse.textrank(sentence, topK=10, withWeight=True, allowPOS=('n', 'nr', 'ns', 'nz'))
# print(keywords)
for i in keywords2:
    print(i[0], i[1])


print(HanLP.segment('萧忆情-死性不改'))
for term in HanLP.segment('萧忆情-死性不改'):
    print('{}\t{}'.format(term.word, term.nature)) # 获取单词与词性2222222


def get_cut_word(cut_res):
    words = list(cut_res)
    print(words)
    return words


if __name__ == '__main__':
    pass
    # get_cut_word(words1)
    # get_cut_word(words2)
    # get_cut_word(words3)
    # get_word_tag(words4)
    # get_word_tag(words5)


print(HanLP.segment('你好，欢迎在Python中调用HanLP的API'))
for term in HanLP.segment('下雨天地面积水'):
    print('{}\t{}'.format(term.word, term.nature)) # 获取单词与词性
