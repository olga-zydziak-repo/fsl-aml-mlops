# trening/ — training experiments outside the pipeline

`trening.ipynb` is the pipeline's training step pulled out into a notebook, so the encoder, the episode
(prototype, distance, loss) and the loop can be edited cell by cell without touching `rung9_hpo.ipynb` or
rebuilding the image. Everything that makes a number comparable stays shared with the pipeline: the frozen
data and its SHA, the `split.json` of the campaign (folds, standardisation, floors, thresholds), the episode
samplers and their seed, and the gate arithmetic (lower 95% CI against the fold's threshold).

A self-check cell trains the same configuration through `fsl.training.loop.train()` and compares the weights
bit for bit; until it prints `OK: notebook = pipeline`, numbers from the notebook are not comparable with the
campaign. Every run appends one row to `wyniki/wyniki_f<fold>.csv` (ignored by git) and, with
`LOG_TO_EXPERIMENTS=True`, becomes a run (`run_type="notebook"`) in the same Vertex experiment as the campaigns,
logged by the same `ExperimentLog` the training container uses.

Rules written into the notebook: state the criterion before the run; select on the validation typologies
and let the test confirm once; one fold is one sample, a winner goes to all five; a candidate graduates only
by becoming an encoder in `src/fsl/models/registry.py` and passing the campaign's gate. A change of distance
or loss in the notebook needs the same change in `fsl/models/protonet.py` before promotion, because the gate
measures with `fast_adapt`.

`sufit.ipynb` is step 0 of the training work: where the ceiling is on each fold's test typologies. It measures, in
the gate's units (balanced accuracy), the floor, 5-shot prototypes on raw features (same 1000 episodes as the gate),
the pipeline's ProtoNet, a label curve for raw-feature prototypes (1 to all labels per typology, pool and control
set disjoint by account group as in the active-learning simulation) and a supervised ceiling (logistic regression,
random forest) with full labels. Headroom = ceiling minus prototypes at 5 labels; the verdict rule is written in the
notebook before the run: >= 5 pp means labels (the analyst loop) are the lever, < 3 pp means the features are.
Results go to `wyniki/sufit.csv` and `wyniki/sufit_f<k>.png`.

`tabpfn.ipynb` tests a foundation model for tables (TabPFN v2, ungated checkpoint, no login) as the per-episode
classifier instead of the prototype, on the gate's own test episodes, against the prototype and a logistic
regression fitted on the same 20 support rows. No training. It needs its own kernel so the `fsl` environment stays
untouched (on Workbench):

    python -m venv ~/tabpfn-env && ~/tabpfn-env/bin/pip install -q "tabpfn>=9" ipykernel pyarrow networkx \
      google-cloud-storage && ~/tabpfn-env/bin/pip install -q -e ~/mlops-meta-learning/fsl-projekt --no-deps \
      && ~/tabpfn-env/bin/python -m ipykernel install --user --name tabpfn --display-name "Python (tabpfn)"

Then open `trening/tabpfn.ipynb` with the `Python (tabpfn)` kernel. Criterion and prediction are in the notebook.

Run all three from this directory with the `fsl` kernel; the notebooks find `configs/pipeline_config_samld.json` by
walking up to the repository root.
