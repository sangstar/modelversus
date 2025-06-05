from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Any

import nltk
from bert_score import score
from datasets import Dataset
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

BERT_MODEL = "microsoft/deberta-xlarge-mnli"

# First-time setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

Prediction = str
Predictions = list[Prediction]

Reference = str
References = list[Reference]

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


def save_bert_traced():
    import torch
    from transformers import AutoModel
    class DebertaWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = AutoModel.from_pretrained(BERT_MODEL, output_hidden_states=True)

        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state  # [B, L, H]

    # A bit janky to put so much arbitrary logic in __init__.py, but this is
    # always meant to run on init, so...
    model = DebertaWrapper()
    model.eval()

    # Dummy input for tracing
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    inputs = tokenizer("hello world", return_tensors="pt")

    # This attempts to make sure all parameters have grad disabled,
    # but TorchScript seems very unwilling to ensure grad is disabled
    # when tch loads it later
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        model = DebertaWrapper()
        model.eval()
        traced = torch.jit.trace(model, (inputs["input_ids"], inputs["attention_mask"]))
        traced.save("deberta_trace.pt")


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


# TODO: This is just for one model, will need to do these for model A and model B
# TODO: Implement a bootstrapped A/B test eventually for even more confidence here
# TODO: It may be pretty hazardous to call get_bert_scores(preds, refs) across many
#       processes; may load the model in to memory multiple times unless it can be shared
#       among the processes somehow (not sure how given the way processes have their own memory
#       space), or if not could share among the threads, just keep in mind GIL may slow things down
def calculate_total_score(preds, golds) -> list[float]:
    from modelversus import get_unified_score
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    pred_word_tokens = [pre_process(text) for text in preds]
    gold_word_tokens = [pre_process(text) for text in golds]

    # Combine and tokenize together
    all_texts = preds + refs
    all_tok = tokenizer(all_texts, return_tensors="pt", padding=True, truncation=True)

    # Now split the batch back into preds and refs
    split = len(preds)
    pred_tok = {k: v[:split] for k, v in all_tok.items()}
    gold_tok = {k: v[split:] for k, v in all_tok.items()}

    pred_input_ids = pred_tok["input_ids"].numpy()
    pred_attention_mask = pred_tok["attention_mask"].numpy()

    gold_input_ids = gold_tok["input_ids"].numpy()
    gold_attention_mask = gold_tok["attention_mask"].numpy()

    scores = get_unified_score(
        pred_word_tokens,
        gold_word_tokens,
        pred_input_ids,
        pred_attention_mask,
        gold_input_ids,
        gold_attention_mask
    )
    print(scores)


if __name__ == "__main__":
    preds = [
        "the cat sat on the mat",  # perfect match
        "a dog barked loudly",  # partial match
        "sat the cat on mat the",  # scrambled
        "the feline rested on the carpet",  # paraphrased
        "completely unrelated sentence",  # no overlap
        "d",  # empty prediction
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
