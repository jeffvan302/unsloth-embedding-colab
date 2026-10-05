# Unsloth Embedding Fine-tuning on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeffvan302/unsloth-embedding-colab/blob/main/embedding_finetune_poc.ipynb)

A proof-of-concept notebook that fine-tunes an embedding model on **your own documentation and vocabulary**, end to end, on a **free Google Colab T4 GPU**.

It takes a set of documents, uses a small local LLM to write synthetic training questions for them, fine-tunes [`Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) with [Unsloth](https://github.com/unslothai/unsloth) + LoRA, and measures retrieval quality before and after training on held-out documents.

The same pipeline is designed to scale up to `Qwen3-Embedding-8B` on larger hardware (see [Scaling up](#scaling-up-to-qwen3-embedding-8b)).

---

## What the notebook does

| Step | Section | What happens |
|---|---|---|
| 1 | Install | Installs Unsloth and pinned dependencies (same recipe as Unsloth's official notebooks) |
| 2 | Configuration | One cell holds every setting you'd change: models, data source, chunk size, training hyper-parameters |
| 3 | Sample corpus | 16 short docs about **Halyard**, a *fictional* warehouse-robotics product full of invented jargon, plus a 17-term glossary. Because the terms are made up, the base model can't already know them — the same situation as your internal docs |
| 4 | Load & chunk | Reads documents and splits them into passages of ~80 words |
| 5 | Synthetic library | `Qwen2.5-3B-Instruct` writes 5 realistic questions per passage (keyword queries, full questions, symptom descriptions). The glossary becomes extra `term → definition` pairs. Everything is cached to `synthetic_pairs.jsonl` so you can review or hand-edit it |
| 6 | Train/test split | 20% of passages are held out entirely; their questions are only used for evaluation |
| 7 | Baseline | Loads the base embedding model and measures Recall@k, MRR@10 and NDCG@10 |
| 8 | Hard negatives | For each training question, finds the wrong passage the base model ranks highest, to make training harder and more useful |
| 9 | Fine-tune | LoRA training with `MultipleNegativesRankingLoss` via Unsloth's `FastSentenceTransformer` |
| 10 | Re-measure | Same metrics after training, plus the questions whose results improved most |
| 11 | `search()` | A minimal retrieval function to try your own queries |
| 12 | Save | Saves the LoRA adapter and a merged 16-bit model; optional push to the Hugging Face Hub |
| 13 | Scale-up notes | What to change for the 8B model on multi-GPU hardware |

## Quick start

1. Click **Open in Colab** above.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**.

With the sample data the whole run takes roughly 10–15 minutes.

## Using your own documents

Everything you're meant to customise is marked **🔌 HOOK** in the notebook.

**Documents** — set `DOCS_SOURCE` in the configuration cell:

| Value | Behaviour |
|---|---|
| `"sample"` | The built-in fictional Halyard docs |
| `"upload"` | Prompts you to upload `.md`, `.txt` or `.pdf` files |
| a folder path | Reads every `.md`, `.txt` and `.pdf` under it, e.g. `"/content/drive/MyDrive/my_docs"` (uncomment the Drive-mount line in section 4 first) |

**Word set / glossary** — edit the `GLOSSARY` dict, or point `GLOSSARY_CSV` at a CSV file:

```csv
term,definition
dwell lock,safety state where a robot freezes in place because two sensors disagree
ghost slot,storage location recorded as full that is physically empty
```

**Keep your results** — Colab deletes working files when the session ends. To keep the trained model and the synthetic library, mount Drive and set `OUTPUT_DIR` and `PAIRS_CACHE` to a Drive folder.

## Hooks

| Hook | Default | Replace it to… |
|---|---|---|
| `load_documents()` | sample corpus, upload, or folder | read from Confluence, SharePoint, a database… (return `{name: text}`) |
| `GLOSSARY` / `GLOSSARY_CSV` | 17 fictional terms | teach your own vocabulary, acronyms and synonyms |
| `generate_queries()` | local `Qwen2.5-3B-Instruct` | use a stronger LLM or a hosted API for better questions (take a list of passages, return a list of question lists) |
| `embed()` | the fine-tuned model | plug the model into your RAG or vector-DB pipeline |
| `search()` | top-k cosine search | prototype retrieval behaviour |

## Key settings

| Setting | Default | Notes |
|---|---|---|
| `BASE_MODEL` | `unsloth/Qwen3-Embedding-0.6B` | Embedding model to fine-tune |
| `GENERATOR_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Writes the synthetic questions; fits a T4 |
| `CHUNK_MAX_WORDS` | 80 | Passage size |
| `QUERIES_PER_CHUNK` | 5 | Synthetic questions per passage |
| `TEST_FRACTION` | 0.2 | Share of passages held out for evaluation |
| `TASK_INSTRUCTION` | retrieval of documentation passages | Qwen3-Embedding's query instruction; used identically in training and search |
| `LORA_RANK` | 32 | LoRA rank (alpha set equal) |
| `EPOCHS` / `BATCH_SIZE` / `LEARNING_RATE` | 4 / 32 / 5e-5 | Training hyper-parameters |

## How it works

**Synthetic training data.** Real query logs are rarely available, so an instruct LLM reads each passage and writes questions it answers. The prompt asks for a mix of short keyword searches, full questions and symptom-style descriptions, and discourages copying phrases from the passage so the model learns meaning rather than word overlap.

**Honest evaluation.** The split is by *passage*, not by question: test questions point at passages the model never saw as a positive during training, while the search corpus at evaluation time still contains every passage.

**Hard negatives.** The base model ranks all training passages for each question; the highest-scoring *wrong* passage becomes that question's negative. Passages scoring above 95% of the correct one are skipped, since they may also be correct (a false negative). Held-out passages are never used as negatives.

**Training objective.** `MultipleNegativesRankingLoss` pulls each question towards its passage and away from its hard negative *and* from every other passage in the batch. The query instruction is applied to questions only (via the trainer's `prompts` argument), and a no-duplicates batch sampler keeps the same passage from appearing twice in a batch.

**Metrics.** Each question has one correct passage. *Recall@k* is the share of questions whose correct passage is in the top k; *MRR@10* averages 1/rank; *NDCG@10* averages 1/log2(rank + 1).

## Outputs

| Path | Contents |
|---|---|
| `synthetic_pairs.jsonl` | The generated question → passage library (one JSON object per line) |
| `halyard_embedding_lora/` | LoRA adapter (small; needs the base model to load) |
| `halyard_embedding_lora_merged/` | Merged 16-bit model, a standard sentence-transformers model |
| `poc_outputs.zip` | Adapter + synthetic library, downloaded to your computer |

Using the merged model elsewhere:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("halyard_embedding_lora_merged")
QUERY_PROMPT = ("Instruct: Given a question about our internal documentation, "
                "retrieve the passage that answers it\nQuery: ")

q = model.encode(["robot stuck flashing amber"], prompt=QUERY_PROMPT, normalize_embeddings=True)
d = model.encode(passages, normalize_embeddings=True)   # documents get no prompt
scores = q @ d.T
```

Always use the same query instruction at search time that was used in training, and re-embed your whole corpus whenever you change models.

## Scaling up to Qwen3-Embedding-8B

The code carries over unchanged; these settings change on, for example, a machine with 2 × 96 GB GPUs:

| Setting | Colab PoC | 2 × 96 GB |
|---|---|---|
| `BASE_MODEL` | `unsloth/Qwen3-Embedding-0.6B` | `Qwen/Qwen3-Embedding-8B` |
| `GENERATOR_MODEL` | Qwen2.5-3B-Instruct | a much stronger local model (30–70B) or a hosted API — question quality is the biggest lever |
| `QUERIES_PER_CHUNK` | 5 | 5–10, across thousands of passages |
| `MAX_SEQ_LEN` | 384 | 512–1024 |
| `BATCH_SIZE` | 32 | 128–512 per GPU; consider `CachedMultipleNegativesRankingLoss` for larger effective batches |
| `LORA_RANK` | 32 | 32–64 in bf16 (no 4-bit quantisation needed) |

LoRA on the 8B model fits on a single 96 GB card. To use both cards with DDP, move the code from section 4 onward into `train.py` and run:

```bash
accelerate launch --num_processes 2 train.py
```

Set `ddp_find_unused_parameters=False` in the training arguments, and pass `gather_across_devices=True` to the loss (available in recent sentence-transformers releases) so each GPU also uses the other GPU's batch as negatives.

## Status and limitations

- **Tested:** every cell is valid Python, and the chunking, data split, hard-negative mining and metrics were run end to end with a stand-in model.
- **Not yet verified on a GPU:** question generation, training and saving. These follow the API used in Unsloth's own Qwen3-Embedding notebook, but the first Colab run is the real test. Please open an issue if a cell fails.
- **Small evaluation set:** the sample corpus yields roughly 30 test questions, so before/after numbers are noisy. Treat them as a check that the pipeline works, not as proof of improvement. A real evaluation needs a few hundred questions, ideally including some written by real users.
- **Dependency pins:** the install cell pins `transformers==4.57.6` and `trl==0.22.2` to match Unsloth's notebooks at the time of writing; these may need updating as Colab's base image changes.

## Credits

- [Unsloth](https://github.com/unslothai/unsloth) — fast LoRA fine-tuning; the install cell is adapted from Unsloth's official notebooks ([LGPL-3.0](https://github.com/unslothai/notebooks))
- [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and [Qwen2.5-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) by the Qwen team
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) — training loop, losses and inference
