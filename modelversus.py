from dataclasses import dataclass
import nltk
import bert_score
import torch
import torch.nn as nn
from enum import Enum

from bert_score import score
from datasets import Dataset
import concurrent.futures
from typing import Protocol, runtime_checkable, Any

Prediction = str
Predictions: list[Prediction]

Reference = str
References: list[Reference]


# TODO: THE POINT OF THIS IS: All of these metrics in a vacuum kind of suck, but with a clever
#       scoring function, we can filter only samples that are close to some threshold and present
#       them to the user to judge


@runtime_checkable
class Model(Protocol):
    def generate(self, prompt: str) -> Prediction: ...


@dataclass
class Challenger:
    model: Model


@runtime_checkable
class Task(Protocol):
    def compare(self, preds: Predictions, refs: References) -> Any: ...


class TaskKind(Enum):
    BERTScore = 1
    METEOR = 2
    Fuzzy = 3
    TokenF1 = 4


@dataclass
class TaskContext:
    a: Challenger
    b: Challenger
    task: Task
    dataset: Dataset


@dataclass
class BERTScore(Task):
    model_type = "microsoft/deberta-xlarge-mnli"

    def compare(self, preds: Predictions, refs: References) -> Any:
        p, r, f1 = score(preds, refs, lang="en",
                         model_type="microsoft/deberta-xlarge-mnli")
        bertscore_f1s = f1.tolist()
        return bertscore_f1s
