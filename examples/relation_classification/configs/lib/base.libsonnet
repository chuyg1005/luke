local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

local extra_tokens = ["<ent>", "<ent2>"];

local tokenizer = {"type": "pretrained_transformer",
                   "model_name": transformers_model_name,
                   "add_special_tokens": true,
                   "tokenizer_kwargs": {"additional_special_tokens": extra_tokens}};

local token_indexers = {
            "tokens": {"type": "pretrained_transformer", "model_name": transformers_model_name,
                       "tokenizer_kwargs": {"additional_special_tokens": extra_tokens}
    }};

{
    "dataset_reader": {
        "type": "relation_classification",
        "dataset": std.extVar("DATASET"),
        "tokenizer": tokenizer,
        "token_indexers": token_indexers
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("VALIDATION_DATA_PATH"),
    "trainer": {
        "cuda_device": std.parseInt(std.extVar("CUDA_DEVICE")),
        "num_epochs": 5,
        "checkpointer": {
            "keep_most_recent_by_count": 1
        },
        "optimizer": {
            "type": "adamw",
            "lr": 1e-5,
            "weight_decay": 0.01,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm.weight",
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
        },
        "learning_rate_scheduler": {
            "type": "custom_linear_with_warmup",
            "warmup_ratio": 0.06
        },
        "num_gradient_accumulation_steps": 4,
        "patience": 3,
        "validation_metric": "+micro_fscore"
    },
    "data_loader": {
        "batch_sampler": {
            "type": "relation_classification",
            "batch_size": 8,
            "shuffle": true,
            "train_mode": std.extVar("TRAIN_MODE")
        },
    },
    "validation_data_loader": {
        "batch_sampler": {
            "type": "relation_classification",
            "batch_size": 128,
            "shuffle": false,
            "mode": "validate"
        },
    },
    "random_seed": std.parseInt(std.extVar("SEED")),
    "numpy_seed": std.parseInt(std.extVar("SEED")),
    "pytorch_seed": std.parseInt(std.extVar("SEED")),
}