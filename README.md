# text-descriptor

## Usage

Setup dataset

```bash
$ ./scripts/setup.sh
```

Load descriptor

```python
from TextProcessor import TextProcessor

# Create an instance
processor = TextProcessor()

# Load model, by default loads spacy GloVe model
processor.load_question()

# Get a question string given the ID from the dataset
question = processor.get_question_text(296747003)

# Get word vectors given the ID from the dataset
descriptors = processor.get_question_vector(296747003)
```
