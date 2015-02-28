
# -*- coding: utf-8 -*-

#   Stdlib
import os
import pdb
from pprint import pformat
from collections import Counter

#   Custom
from .base import detect_lang, detect_chars_noneng, get_features
from .constants_tasks import THIS_DIR


class DetectLanguage(object):
    tags_skip = [
            'ngram',
            'sent',
            'punc',
            'url',
            'symbol',
            'email',
            'ht',
            'hashtag',
            'mention',
        ]
    tags_keep = [
        'w',
    ]

    def __init__(self, *args, **kwargs):
        super(DetectLanguage, self).__init__()
        self.feature_getter = GetSuffixes()

    def skip(self, s, tags_skip=None, tags_keep=None):
        tags_skip = tags_skip or self.tags_skip
        tags_keep = tags_keep or self.tags_keep
        skip_ = None
        if not s:
            skip_ = True
        else:
            for f in tags_keep:
                if s.startswith(f):
                    skip_ = False
                    break
            if skip_ is None:
                for f in tags_skip:
                    if s.startswith(f):
                        skip_ = True
                        break
        return skip_

    def fmt_tokens(self, segments, tags_skip=None, tags_keep=None):
        """
        Provide correct formatting for `base.lang_detect_base.detect_lang`.

        Format: [label]|[text]
        """
        #   Remove the position and tag elements.
        tokens = [
            u"{text}".format(text=seg['text'])
                for seg in segments
                    if not self.skip(seg['name'], tags_skip, tags_keep)
        ]
        tokens = u' '.join(tokens).strip()
        if tokens:
            #   Add dummy label.
            tokens = u'|'.join(['', tokens])
        else:
            #   Use `None` as a sentinel to indicate no content to analyze.
            tokens = None
        return tokens

    def detect_lang(self, content, features=None):
        lang = detect_lang(content, features=features)
        return lang

    def finalize(self):
        pass

    def __call__(self, doc, **kwargs):
        keep_lang = kwargs.get('lang_keep') or set()
        skip_lang = kwargs.get('lang_skip') or set()
        fmt_tokens = self.fmt_tokens
        field_in = kwargs.get('field_in')  or 'current'
        lang_info = doc['text']['language']
        content = fmt_tokens(doc['text'][field_in])
        if content:
            tokens = doc['text'].setdefault('tokens', {})
            tokens_feats = dict([
                (t['text'], t['val'])
                    for t in tokens.get('substrs_by_idx', [])
            ])
            if tokens_feats:
                lang = self.detect_lang(content, tokens_feats)
            else:
                #   Get features and retry
                doc = self.feature_getter(doc, **kwargs)
                tokens_feats = dict([
                    (t['text'], t['val'])
                        for t in tokens.get('substrs_by_idx', [])
                ])
                if tokens_feats:
                    lang = self.detect_lang(content, tokens_feats)
                else:
                    lang = 'unk'
        else:
            lang = 'unk'
        lang_info.setdefault('language', lang)
        return doc



class GetSuffixes(object):
    tags_skip = [
            'ngram',
            'sent',
            'punc',
            'url',
            'symbol',
            'email',
            'ht',
            'hashtag',
            'mention',
        ]
    tags_keep = [
        'w',
    ]

    def __init__(self, *args, **kwargs):
        super(GetSuffixes, self).__init__()

    def skip(self, s, tags_skip=None, tags_keep=None):
        tags_skip = tags_skip or self.tags_skip
        tags_keep = tags_keep or self.tags_keep
        skip_ = None
        if not s:
            skip_ = True
        else:
            for f in tags_keep:
                if s.startswith(f):
                    skip_ = False
                    break
            if skip_ is None:
                for f in tags_skip:
                    if s.startswith(f):
                        skip_ = True
                        break
        return skip_

    def fmt_tokens(self, segments, tags_skip=None, tags_keep=None):
        """
        Provide correct formatting for `base.lang_detect_base.detect_lang`.

        Format: [label]|[text]
        """
        #   Remove the position and tag elements.
        tokens = [
            u"{text}".format(text=seg['text'])
                for seg in segments
                    if not self.skip(seg['name'], tags_skip, tags_keep)
        ]
        tokens = u' '.join(tokens).strip()
        return tokens

    def finalize(self):
        pass

    def __call__(self, doc, **kwargs):
        keep_lang = kwargs.get('lang_keep') or set()
        skip_lang = kwargs.get('lang_skip') or set()
        fmt_tokens = self.fmt_tokens
        field_in = kwargs.get('field_in')  or 'current'
        content = fmt_tokens(doc['text'][field_in])
        tokens = doc['text'].setdefault('tokens', {})
        tokens_feats_by_idx = tokens.setdefault('substrs_by_idx', [])
        tokens_feats_by_text = tokens.setdefault('substrs_by_text', [])
        if content:
            feats_by_idx, feats_by_text = get_features(content)
            #############
            # pdb.set_trace()
            #############
        else:
            feats_by_idx, feats_by_text = None, None
        if feats_by_idx:
            """
            [dict(name='suffix', text=feat, val=val, pos=(0,None)) for feat, val in features.iteritems()]
            """
            tokens_feats_by_idx.extend([
                dict(name='substr', text=feat, val=val, pos=(0,None))
                    for feat, val in feats_by_idx.iteritems()
            ])
            tokens_feats_by_text.extend([
                dict(name='substr', text=feat, val=val, pos=(0,None))
                    for feat, val in feats_by_text.iteritems()
            ])
        return doc



class DetectCharsNonEng(DetectLanguage):
    tags_skip = [
            'ngram',
            'sent',
            'punc',
            'url',
            'symbol',
            'email',
            'ht',
            'hashtag',
            'mention',
            'time',
            'date',
            'num',
        ]
    tags_keep = [
        'w',
    ]

    def __init__(self, *args, **kwargs):
        super(DetectCharsNonEng, self).__init__()

    def skip(self, s):
        tags_skip = self.tags_skip
        tags_keep = self.tags_keep
        skip_ = None
        if not s:
            skip_ = True
        else:
            for f in tags_keep:
                if s.startswith(f):
                    skip_ = False
                    break
            if skip_ is None:
                for f in tags_skip:
                    if s.startswith(f):
                        skip_ = True
                        break
        return skip_

    def fmt_tokens(self, segments):
        tokens = [
            u"{text}".format(text=seg['text'])
                for seg in segments
                    if not self.skip(seg['name'])
        ]
        tokens = u' '.join(tokens)
        if not tokens:
            #   Use `None` as a sentinel to indicate no content to analyze.
            tokens = None
        return tokens

    def detect_lang(self, content):
        prop_ne, _, _, charset = detect_chars_noneng(content)
        if prop_ne > .25:
            #   Greater than 25% non-English characters.
            lang_predict = 'non-en'
            charset = charset
        else:
            lang_predict, charset = None, None
        return lang_predict

    def finalize(self):
        pass

    def __call__(self, doc, **kwargs):
        keep_lang = kwargs.get('lang_keep') or set()
        skip_lang = kwargs.get('lang_skip') or set()
        fmt_tokens = self.fmt_tokens
        field_in = kwargs.get('field_in')  or 'current'
        lang_info = doc['text']['language']
        content = fmt_tokens(doc['text'][field_in])
        if content:
            lang, charset = self.detect_lang(content)
        else:
            lang, charset = None, None
        lang_info.setdefault('language', lang)
        lang_info.setdefault('charset', charset)
        return doc



def test_detect_language():
    d1 = (
        dict(
            text = dict(
                current = [
                    dict(pos=None, name='w', text=u"RT"),
                    dict(pos=None, name='mention', text=u"@LaliBeliebsDemi"),
                    dict(pos=None, name='w', text=u"MY"),
                    dict(pos=None, name='w', text=u"PICTURE"),
                    dict(pos=None, name='w', text=u"WITH"),
                    dict(pos=None, name='mention', text=u"@JUSTINBIEBER"),
                    dict(pos=None, name='w', text=u"WHEN"),
                    dict(pos=None, name='w', text=u"HE"),
                    dict(pos=None, name='w', text=u"BOUGHT"),
                    dict(pos=None, name='w', text=u"ME"),
                    dict(pos=None, name='w', text=u"AN"),
                    dict(pos=None, name='w', text=u"IPHONE"),
                    dict(pos=None, name='w', text=u"OMFG"),
                    dict(pos=None, name='w', text=u"SO"),
                    dict(pos=None, name='w', text=u"UNEXPECTED"),
                    dict(pos=None, name='punc', text=u"."),
                    dict(pos=None, name='w', text=u"I"),
                    dict(pos=None, name='w', text=u"LOVE"),
                    dict(pos=None, name='w', text=u"YOU"),
                    dict(pos=None, name='w', text=u"JU"),
                    dict(pos=None, name='w', text=u"..."),
                    dict(pos=None, name='url', text=u"http://t.co/mTbkc1em1c"),
                ],
                language = dict(
                    twitter = "en",
                    gnip    = "en",
                ),
            ),
            author = dict(
                details = dict(
                    language = ["en"]
                )
            )
        ),
        dict(
            text = dict(
                current = [
                    dict(pos=None, name='w', text=u"RT"),
                    dict(pos=None, name='mention', text=u"@LaliBeliebsDemi"),
                    dict(pos=None, name='w', text=u"MY"),
                    dict(pos=None, name='w', text=u"PICTURE"),
                    dict(pos=None, name='w', text=u"WITH"),
                    dict(pos=None, name='mention', text=u"@JUSTINBIEBER"),
                    dict(pos=None, name='w', text=u"WHEN"),
                    dict(pos=None, name='w', text=u"HE"),
                    dict(pos=None, name='w', text=u"BOUGHT"),
                    dict(pos=None, name='w', text=u"ME"),
                    dict(pos=None, name='w', text=u"AN"),
                    dict(pos=None, name='w', text=u"IPHONE"),
                    dict(pos=None, name='w', text=u"OMFG"),
                    dict(pos=None, name='w', text=u"SO"),
                    dict(pos=None, name='w', text=u"UNEXPECTED"),
                    dict(pos=None, name='punc', text=u"."),
                    dict(pos=None, name='w', text=u"I"),
                    dict(pos=None, name='w', text=u"LOVE"),
                    dict(pos=None, name='w', text=u"YOU"),
                    dict(pos=None, name='w', text=u"JU"),
                    dict(pos=None, name='w', text=u"..."),
                    dict(pos=None, name='url', text=u"http://t.co/mTbkc1em1c"),
                ],
                language = dict(
                    twitter     = "en",
                    gnip        = "en",
                    language    = "en"     #   Inserted by "DetectLanguage()"
                ),
            ),
            author = dict(
                details = dict(
                    language    = ["en"]
                )
            )
        )
    )
    d2 = (
        dict(
            text = dict(
                current = [
                    dict(pos=(0, 2), name="w", text=u"RT"),
                    dict(pos=(3, 13), name="mention", text=u"@iphonearu"),
                    dict(pos=(13, 14), name="punc", text=":"),
                    dict(pos=(15, 19), name="w", text=u"\u91cd\u3044\u30a2\u30d5"),
                    dict(pos=(20, 24), name="w", text=u"\u30ea\u306f\u3068\u3075"),
                    dict(pos=(25, 26), name="punc", text=u"\u3002"),
                    dict(pos=(26, 37), name="hashtag", text=u"#iPhone\u3042\u308b\u3042\u308b"),
                ],
                language = dict(
                    twitter = "ja",
                    gnip    = "en",
                ),
            ),
            author = dict(
                details = dict(
                    language = ["ja"]
                )
            )
        ),
        dict(
            text = dict(
                current = [
                    dict(pos=(0, 2), name="w", text=u"RT"),
                    dict(pos=(0, 2), name="w", text=u"RT"),
                    dict(pos=(3, 13), name="mention", text=u"@iphonearu"),
                    dict(pos=(13, 14), name="punc", text=":"),
                    dict(pos=(15, 19), name="w", text=u"\u91cd\u3044\u30a2\u30d5"),
                    dict(pos=(20, 24), name="w", text=u"\u30ea\u306f\u3068\u3075"),
                    dict(pos=(25, 26), name="punc", text=u"\u3002"),
                    dict(pos=(26, 37), name="hashtag", text=u"#iPhone\u3042\u308b\u3042\u308b"),
                ],
                language = dict(
                    twitter     = "ja",
                    gnip        = "en",
                    language    = "ja",
                ),
            ),
            author = dict(
                details = dict(
                    language = ["ja"]
                )
            )
        ),
    )
    #   Case where we need to use our own detection.
    d3 = (
        dict(
            text = dict(
                current = [
                    dict(pos=(0, 2), name="w", text=u"RT"),
                    dict(pos=(3, 13), name="mention", text=u"@iphonearu"),
                    dict(pos=(13, 14), name="punc", text=":"),
                    dict(pos=(15, 19), name="w", text=u"\u91cd\u3044\u30a2\u30d5"),
                    dict(pos=(20, 24), name="w", text=u"\u30ea\u306f\u3068\u3075"),
                    dict(pos=(25, 26), name="punc", text=u"\u3002"),
                    dict(pos=(26, 37), name="hashtag", text=u"#iPhone\u3042\u308b\u3042\u308b"),
                ],
                language = dict(
                    twitter = "ja",
                    gnip    = "en",
                ),
            ),
            author = dict(
                details = dict(
                    language = ["zh"]
                )
            )
        ),
        dict(
            text = dict(
                current = [
                    dict(pos=(0, 2), name="w", text=u"RT"),
                    dict(pos=(3, 13), name="mention", text=u"@iphonearu"),
                    dict(pos=(13, 14), name="punc", text=":"),
                    dict(pos=(15, 19), name="w", text=u"\u91cd\u3044\u30a2\u30d5"),
                    dict(pos=(20, 24), name="w", text=u"\u30ea\u306f\u3068\u3075"),
                    dict(pos=(25, 26), name="punc", text=u"\u3002"),
                    dict(pos=(26, 37), name="hashtag", text=u"#iPhone\u3042\u308b\u3042\u308b"),
                ],
                language = dict(
                    twitter     = "ja",
                    gnip        = "en",
                    language    = "non-en",
                ),
            ),
            author = dict(
                details = dict(
                    language = ["zh"]
                )
            )
        ),
    )
    data = [d1, d2, d3]
    detect_language = DetectLanguage()
    for doc, expected in data:
        # pdb.set_trace()
        print "ORIG:", pformat(doc)
        print "\tEXPECTED:", pformat(expected)
        res = detect_language(doc)
        print '\tRESULT:', pformat(res)
        assert res == expected
        print
    print 'test_detect_language passed!', '\n'



def test():
    test_detect_language()


if __name__ == '__main__':
    test()
