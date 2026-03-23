"""Checkpoint key normalization (training uses gradient-checkpoint wrappers)."""


def state_dict_for_plain_tinygpt(raw_sd: dict) -> dict:
    """Map `CheckpointBlock` keys (`blocks.*.block.*`) to plain `TinyGPT` keys."""
    out = {}
    for k, v in raw_sd.items():
        if k.startswith("blocks.") and ".block." in k:
            k = k.replace(".block.", ".", 1)
        out[k] = v
    return out


def vocab_size_from_checkpoint(raw_sd: dict) -> int:
    """Infer vocab size from saved embedding (matches training data, not an arbitrary corpus path)."""
    sd = state_dict_for_plain_tinygpt(raw_sd)
    w = sd.get("tok_emb.weight")
    if w is None:
        raise KeyError("checkpoint has no tok_emb.weight; cannot infer vocab_size")
    return int(w.shape[0])
