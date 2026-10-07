# Unsloth Embedding Fine-tuning on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeffvan302/unsloth-embedding-colab/blob/main/embedding_finetune_poc.ipynb)

A proof-of-concept notebook that teaches an embedding model **your own documentation and vocabulary**, on a **free Google Colab T4 GPU**, in **five steps**, and ends with a **before-and-after report**.

It writes synthetic training questions for your documents with a small local LLM, fine-tunes [`Qwen3-Embedding`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) with [Unsloth](https://github.com/unslothai/unsloth) + LoRA, measures search quality on held-out documents before and after, and maps how the model's view of your vocabulary changed.

---

## Quick start

1. Click **Open in Colab** above.
2. **Runtime → Change runtime type → T4 GPU**.
3. Press ▶ on each of the five steps in order (or **Runtime → Run all**).

With the built-in sample, the whole run takes roughly 15–20 minutes, most of it the one-off install and question writing. Each step is a Colab form: its settings are on the right, and its code is hidden (double-click a step to see it).

## The five steps

| Step | What happens | Settings in the step's form |
|---|---|---|
| **▶ 1 · Set up and load your data library** | Installs Unsloth (skipped if already installed), checks the GPU, loads documents and the glossary, and splits documents into passages. | Documents (`sample` / `upload` / `folder`), glossary CSV, base model, query instruction, passage size |
| **▶ 2 · Build the synthetic training library** | A local LLM writes realistic questions for every passage (keyword searches, full questions, and symptom descriptions that never name the term). The glossary becomes extra pairs. Saved as a reviewable `.jsonl` library. | Question writer, questions per passage, rounds, regenerate |
| **▶ 3 · Load the base model, split, measure the starting point** | Holds out passages for validation and test, pairs glossary terms with the passages that use them, loads the base model with an empty LoRA adapter, and scores it on the test questions. | Test / validation share, max sequence length, LoRA rank |
| **▶ 4 · Train** | Snapshots the vocabulary, mines hard negatives, fine-tunes while scoring every epoch, keeps the best epoch, draws the training curve and re-runs the test. Press ▶ again with new settings to retrain from a fresh base model. | Epochs, keep best epoch, learning rate, batch size, auto-detected terms |
| **▶ 5 · Build the report** | Builds `training_report.html` with every chart and table, shows it in the notebook, saves the LoRA adapter, and downloads everything as a zip. | Projection (t-SNE / PCA), save adapter, download zip, copy to a Drive folder |

Optional extras after Step 5: a **search box** to try the trained model, and **save the merged model / push to the Hugging Face Hub**.

## The report

`outputs/training_report.html` is a single self-contained file (it works offline) that explains the run from pre-training to post-training:

- **Results at a glance**: right passage ranked first, MRR@10, terms finding their own document and topic separation, before → after.
- **How this works**: the pipeline in plain language, with this run's numbers and settings.
- **Did search improve?** Held-out test scores, example searches before vs after, and the questions that moved up or down most.
- **How training went**: the per-epoch training curve and the epoch that was kept, with advice (train longer, or add more varied data).
- **Did the vocabulary get organised?** Topic separation scores, a map of your terms per topic before vs after, each term's margin for finding its own document, an animated drift plot and nearest-word associations.
- **Why do this** and **caveats and what next**.

## Your data library

| Part | What it is | Where |
|---|---|---|
| **Documents** | `.md`, `.txt` or `.pdf` files | Step 1: `DOCS_SOURCE` = `upload` to pick files, or `folder` + `DOCS_FOLDER` (Google Drive folders are mounted automatically) |
| **Glossary / word set** | Your terms and their meanings, used as extra training pairs and tracked on the vocabulary map | Step 1: `GLOSSARY_CSV`, a CSV with columns `term,definition` |
| **Topic groups** | Which documents belong together, used to colour the vocabulary map | Step 1 code: `DOC_GROUPS` (default: one topic per file) |
| **Words to watch** | Extra terms for the vocabulary map | Step 1 code: `VOCAB_EXTRA` |
| **Example searches** | Shown before vs after in the report | Step 1 code: `EXAMPLE_QUERIES` (default: a few held-out questions) |

```csv
term,definition
dwell lock,safety state where a robot freezes in place because two sensors disagree
ghost slot,storage location recorded as full that is physically empty
```

The built-in sample describes **Halyard**, a *fictional* warehouse-robotics product full of invented jargon (16 documents, 17 glossary terms). Because the terms are made up, the base model can't already know them, which is the same situation as your internal documents.

To read documents from somewhere else (Confluence, SharePoint, a database), replace `load_documents()` in Step 1; it returns `{name: text}`. To write questions with a hosted LLM, replace `generate_queries()` in Step 2; it takes a list of passages and returns a list of question lists.

## How it works

**Synthetic library.** Few teams have labelled question → answer pairs, so an instruct model plays the user. It writes a mix of keyword searches, full questions and symptom-style descriptions, and is told not to copy phrases from the passage, so the model learns meaning rather than word overlap. Glossary terms are paired with their definitions *and* with the passages that use them.

**Honest evaluation.** The split is by *passage*, not by question, so test questions point at passages the model never trained on, while search still runs over every passage. A separate validation set chooses the best epoch, so the test score isn't flattered by that choice.

**Hard negatives.** For each training question, the wrong passage the base model ranks highest becomes a deliberate trap. Passages from the answer's own document, and passages scoring almost as high as the answer, are skipped, since they may also be correct.

**Training.** `MultipleNegativesRankingLoss` pulls each question towards its passage and away from its hard negative and every other passage in the batch. Only a LoRA adapter (about 1–2% of the weights) is trained. Every epoch is scored on the validation set and the vocabulary, and the best epoch is restored.

**Vocabulary map.** Glossary terms, `VOCAB_EXTRA` and up to `N_AUTO_TERMS` auto-detected phrases (two-word phrases and capitalised names that recur in a few documents) are embedded before and after training. Separation scores are computed in the full embedding space; the 2-D maps are projections, so treat their distances as approximate.

**Metrics.** *Recall@k*: share of questions with the right passage in the top k. *MRR@10*: average of 1 ÷ rank. *NDCG@10*: like MRR with a gentler penalty for lower ranks.

### Train longer, or add data?

The training curve in Step 4 tells you:

- **Still rising at the last epoch:** raise `EPOCHS`. Nothing is lost, since the best epoch is kept.
- **Peaks early, then flat:** more epochs won't help. Add more varied data: more questions or rounds in Step 2, more documents or glossary terms, or a stronger question writer.
- **Jumpy:** lower `LEARNING_RATE` (e.g. `5e-5`).

## Outputs

Everything is written to `outputs/` and downloaded as `embedding_finetune_outputs.zip`:

| Path | Contents |
|---|---|
| `training_report.html` | The report |
| `charts/` | Training curve, vocabulary map, term margins, interactive drift plot |
| `term_metrics.csv` | Per-term before / after numbers |
| `lora_adapter/` | The trained LoRA adapter (small; needs the base model to load) |
| `synthetic_library_*.jsonl` | The question → passage library |
| `merged_model/` | Only if you run the optional "save merged model" extra: a standard sentence-transformers model |

Using the merged model elsewhere:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("outputs/merged_model")
QUERY_PROMPT = ("Instruct: Given a question about our internal documentation, "
                "retrieve the passage that answers it\nQuery: ")

q = model.encode(["robot stuck flashing amber"], prompt=QUERY_PROMPT, normalize_embeddings=True)
d = model.encode(passages, normalize_embeddings=True)   # documents get no prompt
scores = q @ d.T
```

Use the same query instruction at search time as in training, and re-embed your whole corpus whenever you change models.

## Scaling up to Qwen3-Embedding-8B

The same steps carry over. On, for example, 2 × 96 GB GPUs:

| Setting | Colab (T4) | 2 × 96 GB |
|---|---|---|
| `BASE_MODEL` | `unsloth/Qwen3-Embedding-0.6B` | `Qwen/Qwen3-Embedding-8B` |
| `GENERATOR_MODEL` | Qwen2.5-3B-Instruct | a much stronger local model (30–70B) or a hosted API; question quality is the biggest lever |
| `QUERIES_PER_CHUNK` | 8 | 8–10, across thousands of passages |
| `MAX_SEQ_LEN` | 384 | 512–1024 |
| `BATCH_SIZE` | 32 | 128–512 per GPU; consider `CachedMultipleNegativesRankingLoss` for larger effective batches |
| `LORA_RANK` | 32 | 32–64 in bf16 (no 4-bit quantisation needed) |

LoRA on the 8B model fits on a single 96 GB card. To use both cards with DDP, export the code of Steps 1–4 into `train.py` and run `accelerate launch --num_processes 2 train.py`, with `ddp_find_unused_parameters=False` in the training arguments and `gather_across_devices=True` on the loss so each GPU also uses the other GPU's batch as negatives.

## Status and limitations

- **Tested:** every step was run end to end, including re-running Step 4, with a stand-in model on CPU. The report was checked in a browser at desktop and phone widths, in light and dark mode.
- **Not yet verified on a GPU in this five-step layout:** the install, question writing, training and saving code is unchanged from the earlier version, which ran on Colab, but the first run of this layout is the real test. Please open an issue if a step fails.
- **Small evaluation set:** the sample corpus gives about 50 test questions, so before/after numbers are noisy. A real evaluation needs a few hundred, ideally including some written by real users.
- **Dependency pins:** Step 1 pins `transformers==4.57.6` and `trl==0.22.2` to match Unsloth's notebooks at the time of writing; these may need updating as Colab's base image changes.

## Credits

- [Unsloth](https://github.com/unslothai/unsloth): fast LoRA fine-tuning; the install code is adapted from Unsloth's official notebooks ([LGPL-3.0](https://github.com/unslothai/notebooks))
- [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and [Qwen2.5-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) by the Qwen team
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers): training loop, losses and inference
