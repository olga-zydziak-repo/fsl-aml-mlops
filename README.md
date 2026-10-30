# Few-Shot Learning MLOps on Vertex AI

[![ci](https://github.com/olga-zydziak-repo/fsl-aml-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/olga-zydziak-repo/fsl-aml-mlops/actions/workflows/ci.yml)

A gated MLOps cycle for few-shot classifiers on Google Cloud Vertex AI (Pipelines, Custom Jobs, Hyperparameter Tuning, Model Registry, Artifact Registry, Experiments). Data is frozen and checksummed. Gate thresholds are measured against a per-domain floor. The registry and promotion work by moving pointers. Explanations are tested before you are asked to trust them. An active-learning component tells an analyst which case to label next.

The model is deliberately simple: Prototypical Networks. The point of the repository is what surrounds it, the parts that make an ML system auditable rather than just accurate.

Three domains run through the same pipeline: Omniglot (characters), NWPU-RESISC45 (satellite scenes) and SAML-D (anti-money-laundering typologies, tabular). SAML-D is the demo domain.

## Headline: AML typologies on SAML-D

**Active learning halves the labels.** When the model picks the next case to label (margin: the case closest to the boundary between two prototypes), it reaches the accuracy that random labelling gets with 24 labels after 11. The criterion was fixed before the run: margin beats random by +4.2 pp on average over budgets of 8–16 labels, 95% CI [+2.2, +6.1], and the gain is positive on every one of five folds. The same protocol runs as a pipeline component. On the demo fold it reaches the random@24 accuracy at 10 labels, and with 24 labels it scores 0.925 vs 0.831.

**The label queue lands on the model's mistakes, with an exact explanation.** Every case in the demo queue was misclassified at the time it was queued. Most were the same confusion, Layered_Fan_In vs Gather-Scatter. Nearest-prototype classification in raw feature space decomposes exactly into per-feature terms, so the analyst sees which features pulled the case the wrong way.

**The gate rejected a learned metric that adds nothing.** In the 5-fold campaign the trained encoder beat a random untrained encoder by −1.2 pp on average (CI [−10.0, +7.6]), and all five models were rejected. Absolute accuracy was high (0.66–0.91 on 4-way tasks, chance 0.25), but it comes from the case features, not from meta-learning. The pipeline caught this because its threshold measures what learning adds over a floor.

**Scope.** The task is assigning a typology to an already detected case, not detecting laundering, and the data is synthetic. The full chain, controls and caveats are in [`REPORT_SAMLD.md`](REPORT_SAMLD.md) (in Polish). A 15-minute demo script is in [`DEMO.md`](DEMO.md) (in Polish).

## What is being trained

Few-shot classification asks: given K examples each of N classes never seen in training, which one is this? Training is episodic. Each episode samples N classes, K support examples per class and a set of queries. An encoder embeds everything, and each class's support embeddings are averaged into a prototype. Queries go to the nearest prototype by Euclidean distance. Test classes are disjoint from training classes.

The encoder is chosen per domain through a registry: Conv4, ResNet-12 or an ImageNet-pretrained ResNet-18 for images, and an MLP for tables.

Two properties shape the rest of the repository:
- **The decision is a distance comparison.** "Class B, because it sits closest to these examples" is the real mechanism, so explanations can be exact rather than post hoc.
- **Accuracy is a distribution over episodes.** Gates reason about the mean, its 95% confidence interval and the spread, never a single number.

## Domains and results

- **Omniglot**, 5-way 5-shot: about 98%. Hyperparameter tuning showed a flat landscape (1.4 pp spread over 8 trials).
- **RESISC45**, 5-way 5-shot, Conv4: 66.2% baseline and 68.3% with lossless flip + 90° rotation augmentation. That was the winner of a four-policy ablation; TrivialAugment hurt. A pretrained ResNet-18 reached 78.8% in local screening and is not yet through the pipeline.
- **SAML-D, case level**, 4-way 5-shot over 17 suspicious typologies, 1,902 cases and 31 features in 5 class-disjoint folds. The results are above, and the decision chain that led to the case unit is in the report.

## The pipeline

```
verify frozen data (manifest + SHA-256)
  -> resolve hyperparameters (config, saved tuning winner, or a Vizier study)
  -> train [:candidate image, Custom Job]  -- opens the Experiments run: params, curve, result
  -> LIGHT GATE (lower CI >= threshold, std <= max)  -- adds its verdict to that run
       |
       +-- PASS -> REGISTER (new version, no alias)
       |            -> HEAVY EVAL (re-evaluates the SAVED artifact; HTML report)
       |                 +-- PASS and enable_promotion -> PROMOTE model alias + image tag
       |            -> EXPLAIN (HTML report)
       |            -> ACTIVE LEARNING (enable_al; HTML report + label queue; adds its numbers to the run)
       |
       +-- FAIL -> REJECTION REPORT (which criterion failed, by how much)
                   -> EXPLAIN (model_version = "not-registered")
                   -> ACTIVE LEARNING (enable_al; "not-registered")
```

Every run ends in one of two explicit outcomes: a promotion or an explained rejection. Explain and active learning describe the model and never decide anything, so they run in both branches. A rejected model still yields its explanation and a label queue.

**Fold campaign as one run.** A second pipeline loops over the folds with `dsl.ParallelFor`, two at a time. Its first step after verify, `prepare_experiment`, computes the experiment design at run time: which typologies take part, the class-disjoint folds, the per-fold standardisation and the floor and gate threshold of every fold. It writes them as one content-addressed `split.json` and returns the list of folds, and the loop runs over that list, so the number of folds follows from the data. Each fold runs the same steps and reads its threshold and floor from that file. `dsl.Collected` gathers one result row per fold into a campaign summary that computes the per-fold lift over the floor, the mean lift with a t-interval across folds, the gate verdicts and the active-learning savings. It produces one HTML report, one e-mail and one campaign-level Experiments run. The reportable result is computed by the pipeline, not by a notebook cell.

**One Experiments run per trained model.** The training container opens the run (named `<pipeline job>-f<fold>`, in the experiment from the config) and logs the parameters it trained with, the train/val curve as time series (Vertex TensorBoard) and the test result. The gate resumes the same run and adds the threshold, the floor, the lift and its verdict; active learning resumes it and adds its gain over random labelling and the labels it needs to match it. The campaign summary logs one more run, named after the pipeline job, with the mean lift, its interval and the folds passed. So the Experiments table holds everything the campaign e-mail says, one row per model and one per campaign, and two campaigns or two trainings of the same fold are compared by selecting their runs; the curves of the selected runs show side by side in TensorBoard. The three writers never overlap (train → gate → active learning), because the SDK's `log_*` calls are read-modify-write without a version check.

**Notifications, without keys.** The whole pipeline sits inside `dsl.ExitHandler` with Vertex's built-in e-mail component, so every run ends with a status mail and a link. The analyst also gets a content mail. The active-learning step, or the campaign summary, writes one marked line to the logs. A Cloud Monitoring log-based alert extracts its fields (queue, savings, verdicts, link) and mails them. Nothing in the pipeline holds an SMTP password or API key.

**Analyst hand-off in GCS.** Every active-learning step writes one batch to `gs://<bucket>/analyst/do_oznaczenia/<run>/fold<k>/`. The batch holds `kolejka.csv` (one row per queued case: size, period, the two typologies the model is torn between, the margin, the features pulling each way, and an empty `typologia_analityka` column), `transakcje.csv` (the raw transactions of those cases, no labels) and a README. The analyst puts the filled `kolejka.csv` under the same path in `analyst/oznaczone/`. That folder is the input of the analyst round (see Active learning). The case → transaction table is rebuilt once by `samld_case_analyst_data.ipynb`, with the freeze's own linking function, and checked against the frozen case ids.

## Components

Only steps that need the few-shot code (`fsl` package) run in the project's image. Everything else is a lightweight component doing arithmetic and API calls.

- **verify_frozen**: reads the dataset's manifest from GCS and records its SHA-256 in the run. The byte check itself happens in every step that loads data: the loader downloads the frozen archive, recomputes its SHA-256 and stops the step on a mismatch, so nothing trains on anything but the frozen bytes. `scripts/verify_frozen.py` runs the full check (the archive and every file inside) from the command line; the pipeline component does not call it yet.
- **prepare_experiment** (container, campaign only): `scripts/prepare_experiment_entry.py` with `src/fsl/data/split.py`. Drops typologies that cannot form a grouped 5+5 episode, cuts the rest into blocks of n_way typologies (fold k tests block k, validates on block k+1, trains on the rest; leftovers get an extra fold), computes mu/sigma per fold on training typologies only, and the floor of every fold: a random, never-trained MLP, 3 seeds × 400 test episodes, threshold = min(floor + 0.10, floor + half of the floor's errors). Writes `gs://<bucket>/splits/<dataset>/<split_sha>/split.json`; the model's `architecture.json` and the analyst's `stan.json` carry `split_uri` and `split_sha`, the loader refuses a split computed on other data, and an analyst round refuses to mix splits. While the frozen archive still holds its own `folds.json`/`stats.json`, the step compares against them and stops the run before training if folds or statistics differ.
- **resolve_hyperparameters**: passes config values through, reads the saved tuning winner for this dataset, or runs a Vertex Vizier study. Winners are keyed by dataset, and a tuning run worse than the saved winner never overwrites it.
- **train** (container, Custom Job): runs `scripts/train_pipeline_entry.py` with the same `fsl.training.loop.train()` the notebooks use. It opens the Experiments run of this model (parameters, TensorBoard curve, test result) and hands its name downstream. It saves `model.pt` and `architecture.json`, which hold the encoder, episode shape, seed, fold, split and the run name.
- **light gate**: lower CI bound ≥ threshold and std ≤ maximum, on the training-time test numbers. Resumes the training's run and adds threshold, floor, lift and the verdict.
- **register**: a new version of `fsl-protonet-{dataset}` in Model Registry, with metrics and the data SHA. Registering is cataloguing, not promotion.
- **heavy eval** (container): downloads the registered artifact, rebuilds the model from `architecture.json` and recomputes accuracy on fresh test episodes. It checks consistency with training within combined confidence intervals, the lower CI and stability.
- **promote model / promote image**: moves the `production` alias to the verified version and retags the image by digest. Promoted bytes equal tested bytes. `enable_promotion=False` runs everything except the alias move, which is used for ablations and fold campaigns.
- **rejection report**: gauges showing where the model landed relative to each threshold.
- **explain** (container): see below.
- **active learning** (container): see below. Resumes the training's run and adds its headline numbers.

## Thresholds measure learning, not data

A good representation can make an untrained encoder look strong. On SAML-D windows, a random MLP scored the same as nearest-centroid on raw features. A fixed accuracy bar would then reward the features and say nothing about meta-learning.

Each tabular fold therefore gets a floor: a ProtoNet with a random, untrained encoder, averaged over 3 seeds × 400 test episodes. The threshold is the floor plus 0.10, or the floor plus half of its errors, whichever is smaller. The second term keeps a high floor from producing an unreachable bar. The rule was written down before any floor was computed, and the gate checks the lower CI bound against it.

## Explainability

**Images.** A decision-level report: distance bars to every prototype, the support examples of the chosen class, the confident, borderline and wrong decisions, a 2D map of one episode and a histogram of margins.

**Tables.** The report explains the decision margin itself:

    m(x) = ||f(x) - p_contrast|| - ||f(x) - p_chosen||

This is the exact logit gap the model uses. The method is contrastive Integrated Gradients. The contrast is the runner-up for a correct decision and the true class for a mistake ("why B and not A"). The baseline is the contrast class's mean support example. Attributions are rolled up into feature families and shown on case studies in raw units (amounts, days, number of accounts), with each case's id and size. A global view shows per-typology confusion and what each typology is recognised by.

Before the case studies, the report states whether its explanations can be trusted. The tests have thresholds fixed in advance:
- completeness: p95 relative residual ≤ 1%;
- deletion faithfulness: removing the top-5 features moves the margin at least 2× more than removing 5 random ones;
- stability under support resampling: Spearman ≥ 0.7 over 20 resamples;
- baseline sensitivity: reported only.

When a test fails, the report says so and says how to read the cases. Counterfactuals toward a normal class are deliberately not produced: for AML that would be an evasion recipe, not an explanation.

**Active-learning queue.** For nearest-prototype classification in raw features, with A the current answer and B the runner-up,

    d(x, B)^2 - d(x, A)^2 = sum_j [(x_j - B_j)^2 - (x_j - A_j)^2]

so each feature's contribution to "A rather than B" is a closed-form term. No attribution method is needed.

## Active learning

`scripts/al_simulation_entry.py` runs on the model trained in the same run, over its fold's test typologies, which were never seen in training. Cases are split by account group into a pool and an evaluation set, so one launderer is never on both sides. Each typology starts with one label. Each round, a strategy picks one case, an oracle reveals its typology and that prototype is recomputed. Nothing is retrained.

The component compares three strategies on raw features and on the run's embedding: random, margin (validated) and hybrid (typical cases first, then margin; reported only). Its report holds learning curves, the labels each strategy needs to reach random@24, and the label queue with exact explanations. It uses numpy and torch only, needs no keys or external services, and writes a "not applicable" report for image domains.

The analyst labels a whole batch at once, not one case at a time, so the queue is chosen greedily. The first pick is the smallest margin. Then the model's own guess stands in for that label, its prototype moves, and the next smallest margin is taken. Plain top-k by margin piles near-duplicates onto one boundary and drags that prototype. Along with the batch, the component writes `stan.json`: the labelled cases, the pool, the evaluation set and the queue, all as case ids.

**Closing the loop: the analyst round.** `scripts/al_round_entry.py` runs as a one-step pipeline. A Cloud Run function starts it as soon as a filled `kolejka.csv` lands in `analyst/oznaczone/<batch>/`; `al_runda.ipynb` can also start it by hand. The function (`deploy_trigger.sh`, an Eventarc trigger on the bucket, no keys) ignores every other file, including everything the round writes itself. It names the Vertex job after the batch and the file's generation, so a duplicate event cannot start a second run, while a corrected re-upload does. The round does five things:
1. Reads the answers. Spelling is tolerant and Polish Excel's Windows-1250 is accepted. An empty cell means skipped. A typology outside the fold is set aside.
2. Builds candidate prototypes from the labels in use plus the new answers. No training.
3. Gates the candidate on the evaluation set with two rules. Balanced accuracy may not fall more than 2 pp below the prototypes in use, and no single typology may lose more than 10 pp of its evaluation cases (counting at least 3 cases). If it passes, the candidate goes into use (`analyst/wzorce/<run>/fold<k>/produkcja.json`, with every round's candidate kept next to it). If it fails, the old prototypes stay and the answers are held in `stan.json` for a second review.
4. Writes the next batch as `<batch>-r1`, `-r2` and so on.
5. Writes a Polish round report: accuracy before and after, the verdict, the analyst's answers against the model's suggestions, accuracy per round and the next cases. It also prints one marked line that the third alert rule turns into the analyst's mail.

The second rule came from the first real rounds on SAML-D fold 3. Round 1 lifted the prototypes from 80.1% to 86.8%, almost all of it on Layered_Fan_In (54% → 80%). In round 2, correct answers on the Layered_Fan_In / Gather-Scatter boundary traded the two typologies: Gather-Scatter went from 67% to 92%, while Layered_Fan_In fell from 80% to 49% (13 of 41 cases). The balanced change was only −1.7 pp, so the average-only gate let it through. On the synthetic development set, the average-only gate let a single-typology collapse into use in about 7% of rounds. With the per-typology rule that went to zero, at the cost of more rejected rounds (about 3 in 10 instead of 2 in 10) and about 1 pp lower balanced accuracy after three rounds.

The gate guards the prototypes, not each answer. A mean prototype can get worse from correct boundary cases, and wrong answers can slip through when nothing drops past the limits. With an evaluation class of three cases or fewer, one case moves balanced accuracy by 8 pp or more, and the report says so. Every round report shows the per-typology table next to the verdict.

## Design decisions worth stealing

- **Registration is not promotion.** Every plausible candidate is catalogued. Only heavy-verified versions get the alias, and rejected candidates stay visible.
- **Cheap gate, then expensive gate.** The light gate filters on training numbers. The heavy gate judges the saved artifact, because that is what gets promoted, and it catches serialization and eval-mode bugs the light gate cannot see.
- **Test what you promote.** The image is built first as `:candidate`, tested as is, and promoted by retagging the digest.
- **Pointers over copies.** Production is a moving alias on an immutable artifact. Rollback means re-pointing.
- **Containers only where the domain logic lives.** One image, one entry point per container step, no duplicated model code.
- **Reproducibility as a chain.** SHA-checked frozen data → seed and fold stored in the artifact → the same split rebuilt at evaluation → data SHA in the registry description and the experiment log.
- **Thresholds from a floor, fixed before the floor is known.**
- **Criteria before runs.** Every experiment that changed direction on SAML-D had a pass/kill criterion written before it ran. The report records one criterion revised after a dry run on synthetic data, before any real run, and one proposed revision rejected for lack of evidence.
- **Both endings are first-class.** A rejected run produces a rejection report, an explanation, a label queue and an experiment record (the run is opened before the gate decides anything).

## Running the SAML-D demo

1. **Build the image.** Run `scripts/build_image.ipynb`, which tags it `:candidate`. Any change under `scripts/` or `src/` needs a rebuild, because the pipeline runs what is in the image, not what is on disk.
2. **Freeze the cases.** Run `samld_case_freeze.ipynb`:
   - cell 1 builds `samld-case` in GCS from the SAML-D CSV;
   - cell 2 verifies the freeze through the pipeline's own loader and stops if linking glued cases together;
   - cell 3 computes floors and thresholds into `~/data_cache/samld_case_floors.json`. The campaign no longer needs this file (its `prepare_experiment` step computes them in the run); if present, it is passed to that step for comparison only. The single-fold pipeline still takes its threshold from it.
3. **Set up the mails once.** Put the analyst's address under `notifications.analyst_emails` in the config, then run `analyst_alert_setup.ipynb`. It creates three log-based alert rules (queue, campaign, analyst round), and its last cell sends all three test mails.
4. **Submit a run.** In `rung9_hpo.ipynb`, load `configs/pipeline_config_samld.json` and define the pipeline. Then either submit one fold with `enable_al=True` (fold 3 is the demo fold) or run the campaign cell, which runs every fold the `prepare_experiment` step returns (five on SAML-D) in one pipeline run.
5. **Read the reports.** Open the HTML artifacts of the run: heavy eval or rejection report, explain, active learning, and for a campaign the campaign summary. The mails link there.
6. **Close the loop.** Once: run cell 4 of `al_runda.ipynb`, which compiles the round pipeline and stores it in GCS, then `bash deploy_trigger.sh` in Cloud Shell. After that the analyst fills `kolejka.csv` and puts it in `oznaczone/`; for the demo, cell 3 of `al_runda.ipynb` fills in the true typologies. The upload starts the round: answers → prototypes → gate → next batch → mail.
7. **Reproduce the AL criterion** on all five folds with `samld_case_active_learning.ipynb` (optional).
8. **Experiment with the training** in `trening/trening.ipynb`, a directory of its own outside the pipeline. The notebook loads the same freeze and `split.json`, samples the same episodes and reads the same floor and threshold, so its numbers mean what the campaign's mean; the encoder, the episode (distance, loss) and the loop are plain cells to edit. A self-check cell proves the unchanged notebook reproduces `fsl.training.loop.train()` bit for bit. Each run appends a row to `trening/wyniki/` (git-ignored) and, with `LOG_TO_EXPERIMENTS=True`, becomes a run (`run_type="notebook"`) next to the campaign runs. A candidate graduates by becoming an encoder in `fsl/models/registry.py`; the campaign's gate is the final judge.

## Repository layout

```
fsl-projekt/
├── REPORT_SAMLD.md                  # SAML-D results, controls, caveats (PL)
├── DEMO.md                          # demo script (PL)
├── Dockerfile                       # one image: src/fsl + scripts/
├── pyproject.toml
├── configs/
│   ├── pipeline_config.json         # project, bucket, images, gates, training, HPO
│   └── pipeline_config_samld.json
├── scripts/
│   ├── train_pipeline_entry.py      # container: train + save + KFP outputs
│   ├── hpo_train_entry.py           # container: one Vizier trial
│   ├── evaluate_pipeline_entry.py   # container: heavy eval + HTML report
│   ├── explain_pipeline_entry.py    # container: explainability report
│   ├── al_simulation_entry.py       # container: active learning + label queue
│   ├── prepare_experiment_entry.py  # container: folds, standardisation, floors -> split.json
│   ├── al_round_entry.py            # container: analyst round (answers → prototypes → gate → next batch)
│   ├── deploy_trigger.sh            # Cloud Run function: upload to oznaczone/ starts the round
│   ├── bootstrap_fsl_data.py        # freeze a dataset in GCS + SHA-256 manifest
│   ├── verify_frozen.py
│   ├── screen_encoders.py           # local successive-halving encoder screening
│   ├── publish_template.py
│   └── build_image.ipynb
├── src/fsl/
│   ├── config.py                    # TrainConfig (a run == its config)
│   ├── validate.py                  # fail-fast config checks
│   ├── data/                        # registry + omniglot, resisc45, samld loaders; split.py = experiment design
│   ├── models/                      # encoder registry, protonet, conv4, resnet, mlp
│   └── training/loop.py             # train(), evaluate()
├── notebooks/                       # one notebook per rung + SAML-D studies
└── trening/                         # training experiments outside the pipeline (results git-ignored)
```

## Requirements and gotchas

- You need a GCP project with Vertex AI, GCS and Artifact Registry. The pipeline service account needs read/write on the bucket, model upload and alias updates, and `artifactregistry.tags.create/update` on the image repository.
- Vertex experiment names reject underscores, hence `samld-case` with a hyphen; the registry accepts both spellings.
- KFP conditions on pipeline parameters compare explicitly: `dsl.If(enable_al == True)`.
- Vertex caches a step by its spec and inputs, and the image *tag* is part of the spec, not its digest. Rebuilding `:candidate` therefore does not invalidate the cache: a step whose code changed inside the image is still served from the old run. Steps that must run every time (active learning, the campaign summary, anything that notifies) have `set_caching_options(False)`. After changing training or evaluation code, submit with `enable_caching=False`.
- The analyst round starts from the batch's `stan.json`. Batches written by an image older than the round have none, so the first round needs a run on the rebuilt image.
- Check that the CUDA build of torch matches the driver: a cu130 wheel on a 12.4 driver silently falls back to CPU.
- Serving is out of scope. Few-shot inference needs a support set per request, so a real endpoint means a custom serving container. `serving_container_image_uri` points at the training image only because Model Registry requires one at upload; nothing is deployed, and "deploy" in this repository means moving the alias.

## Status and roadmap

Live and verified:
- the full gated pipeline in both endings, promotion in both registries;
- HPO through Vizier;
- explain reports for images and tables;
- the active-learning component on SAML-D;
- the analyst round on SAML-D fold 3: four rounds, 80.1% → 86.8% → round 2 rejected by the per-typology rule → 91.0% → 92.8%; round 4 was started by the trigger, with no manual step.

Next:
- **A way to confirm or drop held answers** (answers from rejected rounds wait in `stan.json`).
- **Daily scoring** of new flagged cases with `produkcja.json`, which is where the analyst's queue should come from in production.
- **More Integrated Gradients steps** (`--ig-steps 200`) once one real artifact confirms completeness drops below 1%.
- **A production pipeline** on the `:production` image with a schedule.
- **Case building in Airflow.** The experiment design already runs in the pipeline (`prepare_experiment`, see Components; built and tested locally on synthetic data against the unchanged freeze, first Vertex run pending). What remains is the data side: linking, account groups and the 31 features as an Airflow job writing a versioned case snapshot, after which the freeze stops writing `folds.json`/`stats.json` and the loader takes the design only from `split.json`. Campaigns are compared only on the same `split_sha`. Folds are evaluation scaffolding; daily scoring does not need them.
- **A real `verify_frozen` step** that calls `scripts/verify_frozen.py` instead of only reading the manifest.
