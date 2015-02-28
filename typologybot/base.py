
# -*- coding: utf-8 -*-

#   Stdlib
import os
import json
import sys
import pdb
import logging
import itertools
from collections import Counter
from bisect import bisect, bisect_left, bisect_right
from pprint import pformat

#   3rd party
import numpy as np
from textslicer.unicode_utils.unicode_utils import (
    CHARS_NONENG, LANG_BY_CHARS_CP, LANG_BY_CHARS_NAME
)

#   Custom
import da   #   Double array trie implementation from LDIG
from constants import THIS_DIR


PATH_DATA = os.path.join(THIS_DIR, '..', 'data')
#   Use the small language model bundled with LDIG
#   (https://github.com/shuyo/ldig)

PATH_MODEL = os.path.join(PATH_DATA, 'lang_det_model.small')
PATH_FEATURES = os.path.join(PATH_MODEL, 'features')
PATH_LABELS = os.path.join(PATH_MODEL, 'labels.json')
PATH_PARAM = os.path.join(PATH_MODEL, 'parameters.npy')
PATH_DOUBLEARRAY = os.path.join(PATH_MODEL, 'doublearray.npz')

TRIE = da.DoubleArray()
TRIE.load(PATH_DOUBLEARRAY)
PARAM = np.load(PATH_PARAM)

def load_labels(path_labels):
    with open(path_labels, 'rbU') as f:
        return json.load(f)
LABELS = load_labels(PATH_LABELS)
LABEL_MAP = dict((x, i) for i, x in enumerate(LABELS))


def detect_lang(text, features=None):
    text_ = text.split('|')[-1]     # Remove true label from text.
    if features is None:
        features, features_by_text = get_features(text, TRIE)
    y, k_true, k_predict, lang_predict = predict(
        text, features, PARAM, LABELS, LABEL_MAP, 0
    )
    return lang_predict


def detect_chars_noneng(text):
    #   !!!!!!!!!!!!
    assert isinstance(text, unicode)
    if not len(text): raise ValueError("No characters to detect!")
    #   !!!!!!!!!!!!
    #   ==========================
    def is_eng(c):
        eng = True
        for s, e in CHARS_NONENG:
            if s <= c <= e:
                eng = False
                break
        return eng
    #   ==========================
    n, n_ne = len(text), 0      #   Total count, non-Eng count
    c_e, c_ne = itertools.tee(text, 2) #   Eng chars, non-Eng chars
    c_e = [ c for idx, c in enumerate(c_e) if is_eng(c) ]
    c_ne = [ c for idx, c in enumerate(c_ne) if not is_eng(c) ]
    prop_ne = len(c_ne) / (n + 0.0)  # Proportion of non-Eng chars
    if prop_ne > .1:
        _, lang = detect_chars_lang(u''.join(c_ne))
    else:
        lang = None
    return prop_ne, c_e, c_ne, lang


def detect_chars_lang(text):
    #   !!!!!!!!!!!!
    assert isinstance(text, unicode)
    if not len(text): raise ValueError("No characters to detect!")
    #   !!!!!!!!!!!!
    #   ==========================
    def get_lang(c):
        cp = ord(c)
        ##############
        # print '\t', c, cp
        # pdb.set_trace()
        ##############
        i = bisect_left(LANG_BY_CHARS_CP, cp) - 1
        lang = LANG_BY_CHARS_NAME[i]
        return lang
    #   ==========================
    langs = Counter()
    for c in text:
        langs[get_lang(c)] += 1
    lang_guess, ct_guess = langs.most_common(1)[0]
    prop_guess = ct_guess / (len(text) + 0.0)
    return prop_guess, lang_guess



def get_features(text, trie=TRIE):
    #   NB: Trie was built using "\u0001" as substring seps.
    feats, feats_by_text = trie.extract_features(u"\u0001" + text + u"\u0001")
    return feats, feats_by_text


def predict(text, features, param=PARAM, labels=LABELS, label_map=LABEL_MAP, i=0):
    """
    Use logistic regression to predict the language
    Modified from ldig.
    """
    label_true, text = text.split('|', 1)
    if label_true not in label_map:
        sys.stderr.write(
            u"WARNING : unknown label '{0}' at {1} in {2} (will skip in the future)\n".format(label_true, i + 1, text)
        )
        label_map[label_true] = -1
    k_true = label_map[label_true]
    y = _predict(param, features)
    k_predict = y.argmax()
    #   Map prediction index onto our labels.
    lang_predict = labels[k_predict]
    if y[k_predict] < 0.6: lang_predict = ""
    #   -----------------------------------------------
    logging.debug("%s\t%s\t%s" % (label_true, lang_predict, text))
    #   -----------------------------------------------
    return y, k_true, k_predict, lang_predict


def _predict(param, events):
    """
    Logistic regression

    Original: ldig
    """
    sum_w = np.dot(param[events.keys(),].T, events.values())
    exp_w = np.exp(sum_w - sum_w.max())
    return exp_w / exp_w.sum()


def likelihood_batch(docs, param, labels, trie=TRIE, options=None):
    """
    Original: ldig
    """
    K = len(labels)
    corrects = np.zeros(K, dtype=int)
    counts = np.zeros(K, dtype=int)
    n_available_data = 0
    log_likely = 0.0
    for idx, doc in enumerate(docs):
        y, k_true, k_predict, predict_lang = predict(
            text, trie, label_map, idx
        )
        if k_true >= 0:
            n_available_data += 1
            log_likely -= np.log(y[k_true])
            counts[k_true] += 1
            if (k_true == k_predict) and y[k_predict] >= 0.6:
                corrects[k_predict] += 1
    if n_available_data > 0:
        log_likely /= n_available_data
        for lbl, crct, cnt in zip(labels, corrects, counts):
            if cnt > 0:
                #   ====================================================
                logging.debug(">    %s = %d / %d = %.2f" % (lbl, crct, cnt, 100.0 * crct / cnt))
                #   ===================================================
        #   =======================================================
        logging.debug("> total = %d / %d = %.2f" % (corrects.sum(), n_available_data, 100.0 * corrects.sum() / n_available_data))
        logging.debug("> average negative log likelihood = %.3f" % log_likely)
        #   =======================================================
    return log_likely



def test_detect_lang():
    data = [
        (
            u"en|RT @LaliBeliebsDemi: MY PICTURE WITH @JUSTINBIEBER WHEN HE BOUGHT ME AN IPHONE OMFG SO UNEXPECTED. I LOVE YOU JU... http://t.co/mTbkc1em1c",
            "en"
        ),
        (
            u"jp|RT @rufuwa: \u604b\u4eba\u7e4b\u304e\u306e\u3067\u304d\u308biPhone\u30b1\u30fc\u30b9\uff57\uff57\uff57\uff57\uff57 http://t.co/XuMdqbiVlE #hotopics http://t.co/qzHy6BFCMO",
            "noneng"
        ),
    ]
    for d, e in data:
        logging.info(detect_lang(d))


def test_detect_chars_noneng():
    data = [
        (
            u"RT @LaliBeliebsDemi: MY PICTURE WITH @JUSTINBIEBER WHEN HE BOUGHT ME AN IPHONE OMFG SO UNEXPECTED. I LOVE YOU JU... http://t.co/mTbkc1em1c",
            (0.0, [u'R', u'T', u' ', u'@', u'L', u'a', u'l', u'i', u'B', u'e', u'l', u'i', u'e', u'b', u's', u'D', u'e', u'm', u'i', u':', u' ', u'M', u'Y', u' ', u'P', u'I', u'C', u'T', u'U', u'R', u'E', u' ', u'W', u'I', u'T', u'H', u' ', u'@', u'J', u'U', u'S', u'T', u'I', u'N', u'B', u'I', u'E', u'B', u'E', u'R', u' ', u'W', u'H', u'E', u'N', u' ', u'H', u'E', u' ', u'B', u'O', u'U', u'G', u'H', u'T', u' ', u'M', u'E', u' ', u'A', u'N', u' ', u'I', u'P', u'H', u'O', u'N', u'E', u' ', u'O', u'M', u'F', u'G', u' ', u'S', u'O', u' ', u'U', u'N', u'E', u'X', u'P', u'E', u'C', u'T', u'E', u'D', u'.', u' ', u'I', u' ', u'L', u'O', u'V', u'E', u' ', u'Y', u'O', u'U', u' ', u'J', u'U', u'.', u'.', u'.', u' ', u'h', u't', u't', u'p', u':', u'/', u'/', u't', u'.', u'c', u'o', u'/', u'm', u'T', u'b', u'k', u'c', u'1', u'e', u'm', u'1', u'c'], [], None),
        ),
        (
            u"RT @rufuwa: \u604b\u4eba\u7e4b\u304e\u306e\u3067\u304d\u308biPhone\u30b1\u30fc\u30b9\uff57\uff57\uff57\uff57\uff57 http://t.co/XuMdqbiVlE #hotopics http://t.co/qzHy6BFCMO",
            (0.12222222222222222, [u'R', u'T', u' ', u'@', u'r', u'u', u'f', u'u', u'w', u'a', u':', u' ', u'i', u'P', u'h', u'o', u'n', u'e', u'\uff57', u'\uff57', u'\uff57', u'\uff57', u'\uff57', u' ', u'h', u't', u't', u'p', u':', u'/', u'/', u't', u'.', u'c', u'o', u'/', u'X', u'u', u'M', u'd', u'q', u'b', u'i', u'V', u'l', u'E', u' ', u'#', u'h', u'o', u't', u'o', u'p', u'i', u'c', u's', u' ', u'h', u't', u't', u'p', u':', u'/', u'/', u't', u'.', u'c', u'o', u'/', u'q', u'z', u'H', u'y', u'6', u'B', u'F', u'C', u'M', u'O'], [u'\u604b', u'\u4eba', u'\u7e4b', u'\u304e', u'\u306e', u'\u3067', u'\u304d', u'\u308b', u'\u30b1', u'\u30fc', u'\u30b9'], 'cjk'),
        ),
    ]
    for doc, expected in data:
        # pdb.set_trace()
        logging.info(u"ORIG: {0}".format(doc))
        print u"ORIG: {0}".format(doc)
        logging.info(u"\tEXPECTED: {0}".format(expected))
        print u"\tEXPECTED: {0}".format(expected)
        res = detect_chars_noneng(doc)
        logging.info(u'\tRESULT: {0}\n'.format(res))
        print u'\tRESULT: {0}\n'.format(res)
        assert res == expected
    logging.info('test_detect_chars_noneng passed!\n')


def test_detect_chars_lang():
    data = [
        (
            u"RT @LaliBeliebsDemi: MY PICTURE WITH @JUSTINBIEBER WHEN HE BOUGHT ME AN IPHONE OMFG SO UNEXPECTED. I LOVE YOU JU... http://t.co/mTbkc1em1c",
            (1.0, 'eng'),
        ),
        (
            u"RT @rufuwa: \u604b\u4eba\u7e4b\u304e\u306e\u3067\u304d\u308biPhone\u30b1\u30fc\u30b9\uff57\uff57\uff57\uff57\uff57 http://t.co/XuMdqbiVlE #hotopics http://t.co/qzHy6BFCMO",
            (0.82222222222222222, 'eng'),
        ),
    ]
    for doc, expected in data:
        # pdb.set_trace()
        logging.info(u"ORIG: {0}".format(doc))
        print u"ORIG: {0}".format(doc)
        logging.info(u"\tEXPECTED: {0}".format(expected))
        print u"\tEXPECTED: {0}".format(expected)
        res = detect_chars_lang(doc)
        logging.info(u'\tRESULT: {0}\n'.format(res))
        print u'\tRESULT: {0}\n'.format(res)
        assert res == expected
    logging.info('test_detect_chars_lang passed!\n')


def test():
    # test_detect_lang()
    test_detect_chars_noneng()
    test_detect_chars_lang()


if __name__ == '__main__':
    logging.basicConfig()
    test()


