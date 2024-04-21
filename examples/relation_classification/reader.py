import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
from allennlp.data import DatasetReader, Instance, Token, TokenIndexer, Tokenizer
from allennlp.data.fields import LabelField, MetadataField, SpanField, TensorField, TextField, ListField
from allennlp.data.samplers import BatchSampler
from transformers.models.luke.tokenization_luke import LukeTokenizer
from transformers.models.mluke.tokenization_mluke import MLukeTokenizer

from examples.utils.util import ENT, ENT2, list_rindex


def parse_kbp37_or_relx_file(path: str):
    with open(path, "r") as f:
        for instance in f.read().strip().split("\n\n"):
            input_line, label = instance.strip().split("\n")
            example_id, sentence = input_line.split("\t")

            # make kbp37 data look like relx
            sentence = sentence.strip('"').strip().replace(" .", ".")

            # replace entity special tokens
            sentence = sentence.replace("<e1>", ENT).replace("</e1>", ENT)
            sentence = sentence.replace("<e2>", ENT2).replace("</e2>", ENT2)

            # we do not need some spaces
            sentence = sentence.replace(f" {ENT} ", f"{ENT} ")
            sentence = sentence.replace(f" {ENT2} ", f"{ENT2} ")
            yield {"example_id": example_id, "sentence": sentence, "label": label}


def parse_tacred_file(path: str):
    if Path(path).suffix != ".json":
        raise ValueError(f"{path} does not seem to be a json file. We currently only supports the json format file.")
    for i, item in enumerate(json.load(open(path, "r"))):
        if type(item) is not list:
            examples = [item]
        else:
            examples = item
        for example in examples:
            tokens = example["token"]
            spans = [
                ((example["subj_start"], ENT), (example["subj_end"] + 1, ENT)),
                ((example["obj_start"], ENT2), (example["obj_end"] + 1, ENT2)),
            ]

            # carefully insert special tokens in a specific order
            spans.sort()
            for i, span in enumerate(spans):
                (start_idx, start_token), (end_idx, end_token) = span
                tokens.insert(end_idx + i * 2, end_token)
                tokens.insert(start_idx + i * 2, start_token)

            sentence = " ".join(tokens)
            # we do not need some spaces
            sentence = sentence.replace(f" {ENT} ", f"{ENT} ")
            sentence = sentence.replace(f" {ENT2} ", f"{ENT2} ")

            yield {"example_id": example["id"], "sentence": sentence, "label": example["relation"]}


@DatasetReader.register("relation_classification")
class RelationClassificationReader(DatasetReader):
    def __init__(
            self,
            dataset: str,
            tokenizer: Tokenizer,
            token_indexers: Dict[str, TokenIndexer],
            use_entity_feature: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)

        if dataset == "kbp37":
            self.parser = parse_kbp37_or_relx_file
        elif dataset in {"tacred", "tacrev", "retacred"}:
            self.parser = parse_tacred_file
        else:
            raise ValueError(f"Valid values: [kbp37, tacred], but we got {dataset}")
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.use_entity_feature = use_entity_feature

        if isinstance(self.tokenizer.tokenizer, (LukeTokenizer, MLukeTokenizer)):
            self.head_entity_id = self.tokenizer.tokenizer.entity_vocab["[MASK]"]
            self.tail_entity_id = self.tokenizer.tokenizer.entity_vocab["[MASK2]"]
        else:
            self.head_entity_id = 1
            self.tail_entity_id = 2

    def text_to_instance(self, sentence: str, label: str = None):
        texts = [t.text for t in self.tokenizer.tokenize(sentence)]
        e1_start_position = texts.index(ENT)
        e1_end_position = list_rindex(texts, ENT)

        e2_start_position = texts.index(ENT2)
        e2_end_position = list_rindex(texts, ENT2)

        tokens = [Token(t) for t in texts]
        text_field = TextField(tokens, token_indexers=self.token_indexers)

        fields = {
            "word_ids": text_field,
            "entity1_span": SpanField(e1_start_position, e1_end_position, text_field),
            "entity2_span": SpanField(e2_start_position, e2_end_position, text_field),
            "input_sentence": MetadataField(sentence),
        }

        if label is not None:
            fields["label"] = LabelField(label)

        if self.use_entity_feature:
            fields["entity_ids"] = TensorField(np.array([self.head_entity_id, self.tail_entity_id]))

        return Instance(fields)

    def _read(self, file_path: str):
        for data in self.parser(file_path):
            yield self.text_to_instance(data["sentence"], data["label"])


@BatchSampler.register("relation_classification")
class RESampler(BatchSampler):
    def __init__(self, mode='train', batch_size=32, shuffle=False, train_mode='default'):
        self.mode = mode
        self.batch_size = batch_size
        self.train_item_size = 13
        self.shuffle = shuffle
        self.train_mode = train_mode.split("@")[0]
        if self.train_mode == 'DataAug' and len(train_mode.split("@")) > 1:
            self.k = int(train_mode.split("@")[1])
        else:
            self.k = 10

    def get_num_batches(self, instances) -> int:
        if self.mode == 'train':
            return math.ceil(len(instances) / self.train_item_size / self.batch_size)
        else:
            return math.ceil(len(instances) / self.batch_size)

    def get_batch_indices(self, instances):
        if self.mode == 'train':
            indices = np.arange(0, len(instances), self.train_item_size)
        else:
            indices = np.arange(0, len(instances))

        if self.shuffle:
            np.random.shuffle(indices)

        extend_indices = None
        extend_indices2 = None

        if self.mode == 'train':
            if self.train_mode in {'default', 'Focal'}:
                pass
            elif self.train_mode == "EntityMask":
                indices += 2
            elif self.train_mode in {'DataAug', 'RDataAug'}:
                offsets = np.random.randint(3, 3 + self.k, size=len(indices))
                extend_indices = indices + offsets
            elif self.train_mode == 'RDrop':
                extend_indices = indices.copy()
            elif self.train_mode in {'DFocal', "PoE"}:
                extend_indices = indices + 1  # 追加上entity-only的样本
            # elif self.train_mode.startswith('MixDebias'):
            elif self.train_mode == 'MixDebias':
                offsets = np.random.randint(3, 3 + self.k, size=len(indices))
                extend_indices = indices + offsets
                # 追加上context-only的样本
                extend_indices2 = indices + 2
            elif self.train_mode == 'Debias':  # 加上context-only的样本
                extend_indices = indices + 2

        # sample 产生一个batch的数据
        for start in range(0, len(indices), self.batch_size):
            item_indices = indices[start: start + self.batch_size]
            if extend_indices is not None:  # 在后面追加上这部分元素
                item_indices = np.concatenate(
                    [item_indices, extend_indices[start: start + self.batch_size]])
                if extend_indices2 is not None:
                    item_indices = np.concatenate(
                        [item_indices, extend_indices2[start: start + self.batch_size]])

            # print(item_indices.shape)
            yield item_indices
