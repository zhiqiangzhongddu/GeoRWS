# GeoRWS: Geometry-Regularized Relational Weight Synthesis

Image-free zero-shot classifier expansion. A frozen image classifier `Phi = head(F(x))` with
seen-class head weights `W_S` is extended to unseen classes without training images: a frozen LLM writes
class-pair affinity descriptions, a frozen text encoder embeds them, and three small MLPs (an adaptive
semantic encoder, its decoder, and a scoring head conditioned on each seen-class weight) turn the pairwise
semantics into mixture coefficients over the seen weights. Training is leave-one-out reconstruction of the
seen weights with a classifier-geometry regulariser (KL to a cosine-neighbourhood teacher built from `W_S`)
and a semantic-consistency regulariser (embedding dropout, symmetric KL). At inference each unseen weight
is synthesised as a mixture of seen weights and injected into the head.

The inherited ICIS baselines (ConSE, COSTA, Sub.Reg., wDAE, WAvg, SMO, ICIS) live in
`src/baselines/` with their numerical behaviour unchanged.

## Repository layout

```
scripts/            CLI entry points (data preparation, descriptions, text embedding, ZSL runs)
src/config/         yacs defaults, one module per block; override with dotted `key value` pairs
src/data_loader/    xlsa17/SDGZSL .mat loading, class names, train/val class split
src/descriptions/   LLM prompts + client + JSONL cache, text encoders + .pt embedding cache
src/georws/         model, losses, trainer, synthesis, perturbations, validation, run
src/baselines/      ICIS-derived baselines
src/zsl/            shared infrastructure: base classifier, REGRESSOR evaluation, dispatch, result rows
src/utils/          paths, seeds, result TSV with file lock
data/, outputs/     datasets, caches and results (git-ignored except small lists and .gitkeep)
```

## Environment

```bash
conda create -n georws -y python=3.10
conda activate georws
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


## Data

CUB, AWA2 and SUN use the ICIS files `att_splits.mat`, `res101.mat` and `res101_finetuned.mat` from the
SDGZSL archive (Google Drive id `1KxFC6T_kGKCNx1JyX2FOaSimA0DOcU_I`; its proposed splits are identical to
xlsa17). The preparation script downloads the archive, places the files under `data/<DATASET>/` and writes
the cleaned class-name lists `data/class_names/<DATASET>.txt`:

```bash
python scripts/run_data_preparation.py
```

Class ids are 0-based positions in `allclasses_names`; labels in the `.mat` files are 1-based and shifted on
load. Seen classes split into the xlsa17 train/val classes (CUB 100/50, AWA2 27/13, SUN 580/65), used only
by the validation mode.

## Pipeline

### 1. Class-pair affinity descriptions

```bash
export OPENAI_API_KEY=...          # or any OpenAI-compatible endpoint via llm.base_url / llm.api_key_env
python scripts/run_descriptions.py dataset CUB llm.name gpt-4o descriptions.kind pairwise
python scripts/run_descriptions.py dataset CUB llm.name gpt-4o descriptions.kind classwise   # ablation input

# dry run without a key
python scripts/run_descriptions.py dataset CUB llm.provider fake descriptions.max_items 20
```

Descriptions are cached in `data/descriptions/<DATASET>/<llm>/{pairwise,classwise}.jsonl` with a manifest
(prompt version, model, sampling parameters, pair mode, class list); generation resumes from the cache and
aborts on a manifest mismatch. Pairs are ordered target-reference pairs by default (all seen-seen and all
unseen-seen pairs: CUB 22,350 + 7,500; AWA2 1,560 + 400; SUN 415,380 + 46,440). `descriptions.symmetric
True` halves the seen-seen count by sharing one description per unordered pair.

### 2. Text embeddings

```bash
python scripts/run_text_embedding.py dataset CUB llm.name gpt-4o text_encoder.name clip   # clip | bert | t5
```

Writes `data/embeddings/<DATASET>/<encoder>/<llm>/{pairwise,classwise}.pt` (dense `[C, C, d]` / `[C, d]`
tensors indexed by class id) and `data/embeddings/<DATASET>/<encoder>/label.pt` ("a photo of a {class}").
Long descriptions are embedded in context-sized chunks pooled by content length; rows are L2-normalised.
`text_encoder.truncation_frac 0.5` additionally writes the truncated file used by the robustness study.

### 3. GeoRWS

```bash
python scripts/run_zsl.py method georws dataset CUB llm.name gpt-4o text_encoder.name clip num_runs 5 device 0
```

The base classifier (linear head on the seen `res101_finetuned` features, as in ICIS) is trained once per
seed and cached under `outputs/base_classifiers/`. Every run appends one row to `outputs/results/zsl.tsv`
with the mean, population std and per-seed values of ZSL accuracy, GZSL unseen / seen accuracy and `H`,
plus the identifying settings. Synthesised weights and the affinity matrix `alpha[u, s]` are saved under
`outputs/georws/<DATASET>/<run_tag>/`.

Main options (`src/config/_georws.py`):

| key | meaning |
| --- | --- |
| `georws.lam`, `georws.beta`, `georws.gamma` | weights of the reconstruction, geometry and consistency losses |
| `georws.tau_w`, `georws.aug_dropout` | teacher temperature, dropout rate of the consistency augmentation |
| `georws.use_geo`, `georws.use_cons` | ablations "w/o L_geo", "w/o L_cons" |
| `georws.semantic_source pairwise|classwise|label` | ablations "w/ class description", "w/o affinity description" |
| `georws.adaptive_encoder False` | ablation "w/o semantic encoding" (`z = a`, no decoder) |
| `georws.unseen_scale seen_norm|none|icis` | deployment rule for the synthesised weights (see below) |
| `zsl.seen_logit_bias` | calibrated stacking: constant subtracted from the seen logits at GZSL evaluation |
| `georws.perturb.kind embedding_dropout|gaussian_noise|text_truncation|pair_swap` | inference-time semantic perturbations (robustness table) |
| `train.lr`, `train.batch_size`, `train.num_epoch` | Adam learning rate, masked classes per batch, epochs (fixed budget) |

**Deployment rule.** A convex mixture of seen weights gives unseen logits that never exceed the largest seen
logit, so injecting the mixture unchanged cannot win a GZSL argmax (`u ~ 0`). The default
`georws.unseen_scale seen_norm` rescales every synthesised row to the mean seen-row norm; `none` injects the
mixture unchanged. The choice is recorded in the result rows.

### Hyper-parameter selection (image-free w.r.t. unseen classes)

```bash
python scripts/run_zsl.py method georws dataset CUB zsl.mode validate georws.beta 0.1 georws.gamma 0.1 georws.tau_w 0.1
```

Validation uses the xlsa17 train/val class split of the seen classes and only `trainval_loc` images: a
validation base head is trained on the train classes (20 % of their images held out per class), GeoRWS
synthesises the val-class weights, and pseudo-ZSL / pseudo-GZSL metrics are computed on the held-out and
val-class images. No test image and no unseen-class description is read. Select `beta, gamma, tau_w,
aug_dropout, unseen_scale, seen_logit_bias` on validation `H`, then run `zsl.mode test`.

### Baselines

```bash
python scripts/run_zsl.py method icis dataset CUB train.cos_sim_loss True train.include_unseen True \
  train.num_layers 2 train.beta1 0.9 train.lr 0.00001 train.batch_size 16 train.embed_dim 2048 \
  zsl.strict_eval True train.early_stopping_slope True zsl.calc_entropy True num_runs 5 device 0
python scripts/run_zsl.py method conse dataset CUB num_runs 5 device 0
python scripts/run_zsl.py method smo dataset CUB zsl.norm_scale_heuristic True baselines.vgse_alpha 0.0 num_runs 5 device 0
```

Methods: `conse`, `costa`, `subreg`, `wdae`, `wavg`, `smo`, `icis`, `georws`. `zsl.class_embedding
att|wiki2vec|cn|clip|llm` selects the baselines' class-wise semantic input (`llm` = the LLM class-wise
descriptions embedded in step 2). `baselines.conse_fixed_eval True` fixes the label mapping of the inherited
ConSE GZSL-unseen evaluation (legacy behaviour is the default).
