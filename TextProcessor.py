"""Text Processor module to handle VQA dataset"""
import json
import spacy
import numpy as np

DEFAULT_VQA_PATH = './dataset/vqa2/v2_OpenEnded_mscoco_train2014_questions.json'

class TextProcessor():
    """Load a NLP model and VQA dataset"""

    def __init__(self):
        self.questions = {}

        # Loads predefined spacy model
        self.nlp = spacy.load('en')

        # Init functions
        self.load_question()

    def load_question(self, vqa_path=DEFAULT_VQA_PATH):
        """Loads a question from VQA dataset

        @param vqa_path (str): relative path to the VQA dataset
        """
        with open(vqa_path) as data_file:
            data = json.load(data_file)

        for obj in data['questions']:
            self.questions[obj[u'question_id']] = obj[u'question']

    def get_question_text(self, question_id, stop_word=False):
        """Get a question text based on the vqa id

        @param question_id (int): VQA question ID
        """
        if not stop_word:
            return self.questions[question_id]
        else:
            question = self.questions[question_id]
            question = self.nlp(question)
            return ' '.join([w.text for w in question if not w.is_stop])

    def get_question_vector(self, question_id, stop_word=False):
        """Get an array of vector based on the vqa id

        The vectors are generated with GloVe model

        @param question_id (int): VQA question ID
        """
        question = self.get_question_text(question_id, stop_word=stop_word)
        tokens = self.nlp(question)
        vectors = np.zeros((len(tokens), 300))

        for i, token in enumerate(tokens):
            vectors[i, :] = token.vector

        return vectors

def main():
    processor = TextProcessor()
    processor.load_question()
    print processor.get_question_text(296747003, stop_word=True)
    print processor.get_question_vector(296747003, stop_word=True)


if __name__ == '__main__':
    main()
