from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Any

import nltk
from bert_score import score
from datasets import Dataset
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import modelversus

# First-time setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

Prediction = str
Predictions = list[Prediction]

Reference = str
References = list[Reference]


# TODO: THE POINT OF THIS IS: All of these metrics in a vacuum kind of suck, but with a clever
#       scoring function, we can filter only samples that are close to some threshold and present
#       them to the user to judge


@runtime_checkable
class Model(Protocol):
    def generate(self, prompt: str) -> Prediction: ...


@dataclass
class Challenger:
    model: Model


@dataclass
class TaskContext:
    a: Challenger
    b: Challenger
    dataset: Dataset


def get_bert_scores(preds: Predictions, refs: References) -> Any:
    p, r, f1 = score(preds, refs, lang="en",
                     model_type="microsoft/deberta-xlarge-mnli")
    bertscore_f1s = f1.tolist()
    return bertscore_f1s


def pre_process(text: str):
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]

    # Stem
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in filtered]
    return " ".join(stemmed)


# TODO: Have separate processes handle these two calculations
#       concurrently
def calculate_total_score(preds: Predictions, refs: References) -> list[float]:
    preds = [pre_process(text) for text in preds]
    refs = [pre_process(text) for text in refs]
    print(preds, refs)

    bert_scores = get_bert_scores(preds, refs)
    word_scores = modelversus.score_batch(preds, refs)
    return [(2 * bert_score + word_score) for bert_score, word_score in zip(bert_scores, word_scores)]


if __name__ == "__main__":
    preds = [
        "the cat sat on the mat",  # perfect match
        "a dog barked loudly",  # partial match
        "sat the cat on mat the",  # scrambled
        "the feline rested on the carpet",  # paraphrased
        "completely unrelated sentence",  # no overlap
        "",  # empty prediction
        "the quick brown fox",  # unrelated but plausible
        "he is eating an apple",  # plausible partial overlap
        "hello world",  # generic short
        "the dog chased the cat",  # role reversal
    ]

    refs = [
        "the cat sat on the mat",  # perfect match
        "the dog barked",  # partial match
        "the cat sat on the mat",  # reference for scrambled
        "the cat sat on the rug",  # semantically close
        "this has no relation at all",  # no overlap
        "the cat sat on the mat",  # non-empty reference
        "a different sentence completely",  # different content
        "she is eating a fruit",  # partial semantic overlap
        "hello there",  # short greeting
        "the cat chased the dog",  # similar, reversed roles
    ]

    print(calculate_total_score(preds, refs))
