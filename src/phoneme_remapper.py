class PhonemeRemapper:
    def __init__(self, *, phonemes, mapping):
        self.phonemes = set(phonemes)
        self.mapping = mapping
        for phonemes in self.mapping.values():
            for phoneme in phonemes:
                assert phoneme in self.phonemes

    def remap(self, phonemes):
        remapped = []
        for phoneme in phonemes:
            if phoneme in self.phonemes:
                remapped.append(phoneme)
            elif phoneme in self.mapping:
                remapped.extend(self.mapping[phoneme])
        return remapped

