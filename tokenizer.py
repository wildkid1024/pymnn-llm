
import base64
import jieba

class Tokenizer:
    def __init__(self, vocab_path="") -> None:
        self._word_decoder = []
        self._word_encoder = {}

        if vocab_path: self.load_vovab(vocab_path)

    def load_vovab(self, vocab_path):
        print(f"load  {vocab_path}... ", end='')
        with open(vocab_path, "r", encoding='utf-8') as vocab_file:
            words = vocab_file.readlines()
            for idx, b64_word in enumerate(words):
                word = base64.b64decode(b64_word).decode()
                self._word_decoder.append(word)
                self._word_encoder[word] = idx
            print("Done!")

    def decode(self, input_ids):
        words = ""
        for input_id in input_ids:
            word = self._word_decoder[input_id]
            if len(word) == 6 and word.startswith('<0x'):
                word = word[1:-1].decode()
            words += word
        return words
    
    def encode(self, input_str):
        ids = []
        words = jieba.cut(input_str, HMM=True)
        for word in words:
            id = self._word_encoder.get(word, -1)
            if id >= 0: ids.append(id)
        return ids


class ChatglmTokenizer(Tokenizer):
    def encode(self, input_str):
        # input_str = '\n' + input_str
        input_ids = super().encode(input_str)
        return input_ids