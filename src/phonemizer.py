import epitran
import epitran.vector
import panphon.distance
import phonemizer.backend
import phonemizer.separator
import re
from abc import ABC, abstractmethod


class Matcher(ABC):
    @abstractmethod
    def match(self, phoneme1, phoneme2):
        pass


class Phonemizer(ABC):
    @abstractmethod
    def phonemize(self, language, sentences):
        pass


class EspeakPhonemizer(Phonemizer):
    def __init__(self):
        self.remap = {
            "en": "en-us",
            "ru": "ru"
        }
        self.espeak = dict()

    def phonemize(self, language, sentences):
        if language not in self.espeak:
            self.espeak[language] = phonemizer.backend.BACKENDS["espeak"](self.remap[language])
        results = self.espeak.phonemize(
            sentences,
            separator=phonemizer.separator.Separator(phone='|'),
            strip=True,
        )
        ret = []
        for sentence, result in zip(sentences, results):
            words = sentence.split(" ")
            result_words = result.split(" ")
            phonemes = []
            if len(words) == len(result_words):
                for word, result_word in zip(words, result_words):
                    phonemes += result_word.split("|")
                    phonemes += [" "]
                if len(phonemes) > 0:
                    phonemes = phonemes[:-1]
            ret.append(phonemes)
        return ret


class EpitranPhonemizer(Phonemizer, Matcher):
    def __init__(self):
        self.remap = {
            "en": "eng-Latn",
            "ru": "rus-Cyrl"
        }
        self.epitran = dict()
        self.vecor_computers = dict()
        self.vectors = dict()

    def phonemize(self, language, sentences):
        if language not in self.epitran:
            self.epitran[language] = epitran.Epitran(self.remap[language])

        ret = []
        for sentence in sentences:
            sentence = re.sub(r"^\s+", "", sentence)
            sentence = re.sub(r"\s+$", "", sentence)
            sentence = re.sub(r"\s+", " ", sentence)
            sentence = re.sub('[\,\?\.\!\-\;\:\"\“\%\‘\”\�]', '', sentence).lower()

            words = []

            for word in sentence.split(" "):
                phonemes = []
                for _, _, _, phoneme, _ in self.epitran[language].word_to_tuples(word):
                    if phoneme == "":
                        continue
                    if phoneme not in self.vectors:
                        if language not in self.vecor_computers:
                            self.vecor_computers[language] = epitran.vector.VectorsWithIPASpace(self.remap[language], ['eng-Latn'])
                        vector_result = self.vecor_computers[language].word_to_segs(sentence)
                        for symbol in vector_result:
                            if symbol[3] == phoneme:
                                self.vectors[phoneme] = list(symbol[5])
                                phonemes.append(phoneme)
                                break
                    else:
                        phonemes.append(phoneme)
                words.append(phonemes)
            ret.append(words)

        return ret

    def match(self, phoneme1, phoneme2):
        return sum(map(lambda x, y: x * y, self.vectors[phoneme1], self.vectors[phoneme2]))


class PanphonMatcher(Matcher):
    def match(self, phoneme1, phoneme2):
        return -panphon.distance.Distance().feature_edit_distance(phoneme1, phoneme2)

