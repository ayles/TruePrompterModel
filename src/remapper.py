import json


class PhonemeRemapper:
    def __init__(self, *, vocabulary, mapping):
        self.vocabulary = set(vocabulary)
        self.mapping = mapping
        for phonemes in self.mapping.values():
            for phoneme in phonemes:
                assert phoneme in self.vocabulary

    def remap(self, phonemes):
        remapped = []
        for phoneme in phonemes:
            if phoneme in self.vocabulary:
                remapped.append(phoneme)
            elif phoneme in self.mapping:
                remapped.extend(self.mapping[phoneme])
        return remapped

    def save(self, stream):
        json.dump({
            "vocabulary": list(self.vocabulary),
            "mapping": self.mapping,
        }, stream, ensure_ascii=False)

    @staticmethod
    def load(stream):
        config = json.load(stream)
        return PhonemeRemapper(vocabulary=config["vocabulary"], mapping=config["mapping"])


