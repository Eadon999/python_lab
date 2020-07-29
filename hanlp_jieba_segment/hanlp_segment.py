from pyhanlp import *


class HanlpSegment:
    StandardTokenizer = JClass("com.hankcs.hanlp.seg.Segment").enableCustomDictionaryForcing  # force custom dict first

    def segment_sentence(self, sentence):
        seg_list = HanLP.segment(sentence)
        return seg_list

    def parse_segment_result(self, seg_result):
        seg_word = ''
        word_tag = ''
        for item in seg_result:
            seg_word = item.word
            word_tag = item.nature
        return seg_word, word_tag


if __name__ == '__main__':
    sentence = '这就是陈奕迅'
    segmenter = HanlpSegment()
    seg_list = segmenter.segment_sentence(sentence)
    print(seg_list)
    word, tag = segmenter.parse_segment_result(seg_list)
    print(word, tag)
