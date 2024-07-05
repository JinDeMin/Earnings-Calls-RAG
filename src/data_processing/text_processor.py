from spacy.lang.en import English

class TextProcessor:
    def __init__(self):
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")

    def split_into_sentences(self, text):
        doc = self.nlp(text)
        return [str(sent) for sent in doc.sents]

    def chunk_sentences(self, sentences, chunk_size=20):
        return [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]