# Data Notes

This repository does not track the real hand image dataset or the full GPT-5 generated annotations.

Local-only files:

- `data/images/`
- `data/palmistry_llava.json`
- `data/palmistry_llava.bak.json`
- `data/test_palmdata.json`
- `data/test_palmdata_backup.json`

## Expected Training Format

The palmistry SFT dataset follows the LLaVA single-image format:

```json
[
  {
    "id": "sample_hand_001",
    "image": "sample_hand_001.png",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n请根据这张手掌图像生成结构化手相分析。"
      },
      {
        "from": "gpt",
        "value": "{\"palmistry_analysis\": { ... }}"
      }
    ]
  }
]
```

See the tracked example:

- [data/examples/palmistry_llava.sample.json](examples/palmistry_llava.sample.json)

## Privacy Recommendation

If the hand images come from real people, do not upload them to a public repository unless you have explicit permission and a clear privacy policy.
