local transformers_model_name = std.extVar("TRANSFORMERS_MODEL_NAME");

{
    "type": "relation_classifier",
    "feature_extractor": {
        "type": "entity",
        "embedder": {
            "type": "transformers-luke",
            "model_name": transformers_model_name,
            "output_embeddings": "entity",
            "use_entity_aware_attention": true,

        }
    },
    "train_mode": std.extVar("TRAIN_MODE"),
}