import os
from typing import Dict, List

import torch
import torch.nn as nn
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy

from .metrics.multiway_f1 import MultiwayF1
from .modules.feature_extractor import RCFeatureExtractor

import torch.nn.functional as F
import numpy as np
import json


@Model.register("relation_classifier")
class RelationClassifier(Model):
    """
    Model based on
    ``Matching the Blanks: Distributional Similarity for Relation Learning``
    (https://www.aclweb.org/anthology/P19-1279/)
    """

    def __init__(
            self,
            vocab: Vocabulary,
            feature_extractor: RCFeatureExtractor,
            dropout: float = 0.1,
            label_name_space: str = "labels",
            text_field_key: str = "tokens",
            ignored_labels: List[str] = None,
            train_mode: str = 'default',
            # validation_data_path: str = None,
    ):

        super().__init__(vocab=vocab)
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(self.feature_extractor.get_output_dim(), vocab.get_vocab_size(label_name_space))

        self.text_field_key = text_field_key
        self.label_name_space = label_name_space

        self.dropout = nn.Dropout(p=dropout)
        # self.criterion = nn.CrossEntropyLoss()

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }
        self.f1_score = MultiwayF1(ignored_labels=ignored_labels)

        # self.train_mode = train_mode

        self.results_save_path = None
        # print(self.vocab.get_token_to_index_vocabulary("labels"))
        args = train_mode.split('@')
        self.train_mode = args[0]
        self.kl_weight = 0.5
        self.lamb = 0.
        # if args[0] == 'MixDebias' and len(args) > 1:
        if len(args) > 1:
            if self.train_mode in {'Debias'}:
                self.lamb = float(args[1])
            elif self.train_mode in {'RDrop', 'RDataAug'}:
                self.kl_weight = float(args[1])
            elif self.train_mode == 'MixDebias':
                self.kl_weight = float(args[1])
                self.lamb = float(args[2])

        pred_root = os.environ.get("PRED_ROOT", None)
        if pred_root is not None:
            dataset = os.environ.get("DATASET", None)
            mode = os.environ.get("TRAIN_MODE", None)
            seed = os.environ.get("SEED", None)
            validation_data_path = os.environ.get("VALIDATION_DATA_PATH", None)
            assert dataset is not None and mode is not None and seed is not None and validation_data_path is not None

            self.save_dir = os.path.join(pred_root, 'luke', dataset, f"{mode}-{seed}")
            os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, "label2id.json"), "w") as f:
                json.dump(self.vocab.get_token_to_index_vocabulary("labels"), f, indent=4)

            val_name = os.path.basename(validation_data_path).replace(".json", "")
            self.results_save_path = os.path.join(self.save_dir, f"{val_name}.txt")
            print(f"model predictions will be saved to {self.results_save_path}")
            if os.path.exists(self.results_save_path):
                os.remove(self.results_save_path)

    def compute_loss(self, logits, labels):
        if not self.training:
            return F.cross_entropy(logits, labels)
        loss = None
        if self.train_mode in {'default', "EntityMask", "DataAug"}:
            loss = F.cross_entropy(logits, labels)
        elif self.train_mode.startswith("MixDebias"):  # our methods
            # print(f"logits: {logits.shape}, labels: {labels.shape}")
            logits_org, logits_aug, logits_co = torch.chunk(logits, 3, dim=0)
            labels_org, labels_aug, labels_co = torch.chunk(labels, 3, dim=0)
            assert torch.equal(labels_org, labels_aug) and torch.equal(labels_org, labels_co)
            probs_org = F.softmax(logits_org, dim=-1)
            probs_co = F.softmax(logits_co, dim=-1).detach()
            label_probs_org = torch.gather(probs_org, dim=1, index=labels_org.unsqueeze(1)).squeeze(1)
            label_probs_co = torch.gather(probs_co, dim=1, index=labels_co.unsqueeze(1)).squeeze(1)
            biased_prob = label_probs_org - self.lamb * label_probs_co
            weights = torch.pow(1 - biased_prob, 2)
            losses = F.cross_entropy(logits_org, labels_org, reduction='none')
            loss = torch.dot(losses, weights) / labels_org.numel()  # debiased loss

            regular_loss = (F.kl_div(F.log_softmax(logits_org, dim=-1), F.softmax(logits_aug, dim=-1),
                                     reduction="batchmean")
                            + F.kl_div(F.log_softmax(logits_aug, dim=-1), F.softmax(logits_org, dim=-1),
                                       reduction="batchmean")) * self.kl_weight
            loss += regular_loss
        elif self.train_mode == 'Focal':
            losses = F.cross_entropy(logits, labels, reduction='none')
            probs = F.softmax(logits, dim=-1)
            label_probs = torch.gather(probs, dim=1, index=labels.unsqueeze(1)).squeeze(1)
            weights = torch.pow(1 - label_probs, 2)
            loss = torch.dot(losses, weights) / labels.numel()  # focal loss
        elif self.train_mode == 'Debias':
            logits1, logits2 = torch.chunk(logits, 2, dim=0)
            labels1, labels2 = torch.chunk(labels, 2, dim=0)
            assert torch.equal(labels1, labels2)
            # compute weights
            probs1 = F.softmax(logits1, dim=-1)  # not detach
            probs2 = F.softmax(logits2, dim=-1).detach()
            label_probs1 = torch.gather(probs1, dim=1, index=labels1.unsqueeze(1)).squeeze(1)
            label_probs2 = torch.gather(probs2, dim=1, index=labels2.unsqueeze(1)).squeeze(1)
            biased_probs = label_probs1 - self.lamb * label_probs2
            # weights = F.sigmoid(biased_probs)
            weights = torch.pow(1 - biased_probs, 2)
            losses = F.cross_entropy(logits1, labels1, reduction='none')
            loss = torch.dot(losses, weights) / labels1.numel()  # debiased loss
        elif self.train_mode in {'DFocal', 'PoE'}:
            logits1, logits2 = torch.chunk(logits, 2, dim=0)
            labels1, labels2 = torch.chunk(labels, 2, dim=0)
            assert torch.equal(labels1, labels2)
            # debiased focal loss
            # if torch.equal(labels1, labels2):
            if self.train_mode == 'DFocal':
                losses = F.cross_entropy(logits1, labels1, reduction='none')
                probs2 = F.softmax(logits2, dim=-1).detach()
                label_probs2 = torch.gather(probs2, dim=1, index=labels2.unsqueeze(1)).squeeze(1)
                weights = torch.pow(1 - label_probs2, 2)
                loss = torch.dot(losses, weights) / labels.numel()  # focal loss
            elif self.train_mode == 'PoE':
                probs1 = F.softmax(logits1, dim=-1)
                probs2 = F.softmax(logits2, dim=-1).detach()
                probs = probs1 * probs2
                loss = F.cross_entropy(torch.log(probs), labels1)

        elif self.train_mode in {'RDrop', 'RDataAug'}:
            logits1, logits2 = torch.chunk(logits, 2, dim=0)
            labels1, labels2 = torch.chunk(labels, 2, dim=0)
            assert torch.equal(labels1, labels2)
            loss = F.cross_entropy(logits1, labels1)
            regular_loss = (F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1),
                                     reduction="batchmean")
                            + F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1),
                                       reduction="batchmean")) * self.kl_weight
            loss += regular_loss

        return loss

    def forward(
            self,
            word_ids: TextFieldTensors,
            entity1_span: torch.LongTensor,
            entity2_span: torch.LongTensor,
            label: torch.LongTensor = None,
            entity_ids: torch.LongTensor = None,
            input_sentence: List[str] = None,
            **kwargs,
    ):
        feature_vector = self.feature_extractor(word_ids[self.text_field_key], entity1_span, entity2_span, entity_ids)
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)
        probs = F.softmax(logits, dim=-1)
        # print(self.training)
        prediction_logits, prediction = logits.max(dim=-1)

        output_dict = {
            "input": input_sentence,
            "prediction": prediction,
        }

        if label is not None:
            # save results
            # print(probs.shape, label.shape)
            # 保存probs
            results = torch.cat([probs, label.unsqueeze(1).float()], dim=1).detach().cpu().numpy()
            if self.results_save_path is not None and not self.training:
                with open(self.results_save_path, "a") as f:
                    np.savetxt(f, results, fmt="%.4f", delimiter=",")

            output_dict["loss"] = self.compute_loss(logits, label)
            output_dict["gold_label"] = label
            self.metrics["accuracy"](logits, label)

            prediction_labels = [
                self.vocab.get_token_from_index(i, namespace=self.label_name_space) for i in prediction.tolist()
            ]
            gold_labels = [self.vocab.get_token_from_index(i, namespace=self.label_name_space) for i in label.tolist()]
            self.f1_score(prediction, label, prediction_labels, gold_labels)

        return output_dict

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
        output_dict["prediction"] = self.make_label_human_readable(output_dict["prediction"])

        if "gold_label" in output_dict:
            output_dict["gold_label"] = self.make_label_human_readable(output_dict["gold_label"])
        return output_dict

    def make_label_human_readable(self, label: torch.Tensor):
        return [self.vocab.get_token_from_index(i.item(), namespace=self.label_name_space) for i in label]

    def get_metrics(self, reset: bool = False):
        output_dict = {k: metric.get_metric(reset=reset) for k, metric in self.metrics.items()}
        output_dict.update(self.f1_score.get_metric(reset))
        return output_dict
