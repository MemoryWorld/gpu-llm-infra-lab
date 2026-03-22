"""Checkpoint key normalization (training uses gradient-checkpoint wrappers)."""


def state_dict_for_plain_tinygpt(raw_sd: dict) -> dict:
    """Map `CheckpointBlock` keys (`blocks.*.block.*`) to plain `TinyGPT` keys."""
    out = {}
    for k, v in raw_sd.items():
        if k.startswith("blocks.") and ".block." in k:
            k = k.replace(".block.", ".", 1)
        out[k] = v
    return out
