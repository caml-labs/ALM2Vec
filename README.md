<h1 align="center">ALM2Vec: Learning Audio Embeddings for Universal Audio Retrieval with Large Audio-Language Models</h1>

<p align="center">
  📄 <strong><a href="https://arxiv.org/abs/2606.30682">Paper</a></strong> |
  🌐 <strong><a href="https://caml-labs.github.io/ALM2Vec">Project Page</a></strong> |
  🤗 <strong><a href="https://huggingface.co/collections/cara-ai/alm2vec">Model</a></strong>
</p>

---

## 📢 News

- **[2026-06-17]** We release the ALM2Vec codebase, project page, and Hugging Face models ([pretrain](https://huggingface.co/cara-ai/ALM2Vec-PT) / [finetune](https://huggingface.co/cara-ai/ALM2Vec-FT)).
- **[2026-07-02]** We release the ALM2Vec technical report on [arXiv](https://arxiv.org/abs/2606.30682).

---

## Requirements

- Python 3.10+
- GPU with ~31GB VRAM

## Quick Start

Install dependencies:

```bash
pip install "transformers>=4.52" torch safetensors torchaudio
```

Example snippet:

```python
import torch
from transformers import AutoModel, AutoTokenizer

# switch between pretrain and finetune
repo_id = "cara-ai/ALM2Vec-PT"
# repo_id = "cara-ai/ALM2Vec-FT"

tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True, torch_dtype=torch.float32).cuda()
model.eval()


QUERY_INSTRUCTION = "Based on the question asked in the text query and context in the audio query, retrieve the relevant text document associated with that question."
DOC_INSTRUCTION = "Represent the user's input."


query_text = [
    "What is the gender of speaker in this audio?",
    "What is the gender of speaker in this audio?",
    "What is the gender of speaker in this audio?",
    "What is the gender of speaker in this audio?",
]
query_audio = ["./assets/example/en_male_music.wav", "./assets/example/en_male.wav", 
               "./assets/example/en_female_music.wav", "./assets/example/en_female.wav"]

doc_text = [
    "male",
    "female",
]

query_embeddings = model.encode(
    text=query_text,
    audio=query_audio,
    task="query",
    instruction=QUERY_INSTRUCTION,
    normalize=True,
    device="cuda",
)
doc_embeddings = model.encode(
    text=doc_text,
    task="document",
    instruction=DOC_INSTRUCTION,
    normalize=True,
    device="cuda",
)

similarity = query_embeddings @ doc_embeddings.T
print(similarity)

similarity = query_embeddings @ query_embeddings.T
print(similarity)

```

## Citation

If you find this work useful, please consider contributing to this repo and cite this work:

```
@misc{lu2026alm2veclearningaudioembeddings,
      title={ALM2Vec: Learning Audio Embeddings for Universal Audio Retrieval with Large Audio-Language Models}, 
      author={Fengjie Lu and Chenang Jiang and Jiarui Hai and Helin Wang and Aaron Yee},
      year={2026},
      eprint={2606.30682},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2606.30682}, 
}
```

## License

The Repository is licensed under CC-BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International).

## Acknowledgement

ALM2Vec is built on [MiDashengLM](https://github.com/xiaomi-research/dasheng-lm) and further trained for universal audio retrieval. We thank MiDashengLM and its underlying [Dasheng](https://github.com/RicherMans/Dasheng) audio encoder for their open-source contributions.

## 🌟 Like This Project?

If you find this repo helpful or interesting, consider dropping a ⭐ — it really helps and means a lot!
