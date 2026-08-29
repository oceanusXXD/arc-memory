# R2W current-paper implementation (v4)

This package contains the corrected implementation of the current R2W manuscript. Its only public API is `r2w`.

## Current algorithm invariants

The current implementation locks the following semantics:

- Core experiment action set is exactly `none / raw / sum / raw+kv / raw+event / raw+hq`. The 11-action family is opt-in only through `use_extended_actions=True`.
- `none` and `raw` are always retained in the full decision stage.
- One structured candidate call produces one `CandidateBundle` containing `summary/kv/event/hq` (and `graph` only in extended mode). Actions reuse this object; selected actions are never regenerated.
- Candidate contract violations reject the whole action. No truncation-and-continue path is used.
- Retrieval is **parent-first**: each sparse/dense channel takes `max(key score)` per `parent_id`, ranks parents, then performs parent-level RRF. A parent occupies at most one top-k slot; cards are not sent to the reader.
- Offline replay uses `none/raw/action` on one local replacement background, hashes the complete reader request, assigns exact zero difference to identical requests, and performs shared Bernoulli query sampling with positive inclusion probability followed by paired Horvitz-Thompson estimates for `DeltaE` and `DeltaF`.
- Evidence can increase query sampling probability but can never make an unlabelled query probability zero.
- Resource accounting is split into `select / commit / index / read`. `select` is excluded from stage-2 arm ranking but included in stage-1 entry value and full path cost.
- The scalar ledger units (`select_scalar`, `commit_scalar`, `index_scalar`, `read_scalar`) and all four resource prices are part of `R2WConfig`, so they enter its immutable experiment hash.
- The value model is parameterised as `DeltaE_raw + DeltaF_action`; it does not independently learn incompatible absolute effects for every action.
- Stage 1 predicts action-specific post-selection values and applies the public selection-cost lower bound. Stage 2 can still return `none`; generated actions must have positive absolute optimistic value and positive conservative increment over `raw`.
- HQ keys are filtered by an explicit QA checker. Generated candidates require an explicit support model; no silent rule-only fallback is used for formal experiments.
- Equivalent actions are merged (for example, `raw+hq` after every HQ key is rejected).
- LoCoMo main folds use natural ID order and the fixed 5 x (6 train / 2 val / 2 test) protocol.
- Online commit is exposed as an atomic parent-object write. Cold archive is not part of the default `none` semantics.
- If the value model is unavailable, the versioned `value_model_fallback` (`raw` by default, or `none`) is used and logged; it never silently substitutes a different model.

## Package layout

`r2w/actions.py` — action sets, one-call candidate bundle, executable contracts.

`r2w/retrieval.py` — parent-level sparse/dense aggregation and RRF.

`r2w/replay.py` — complete request signatures, shared Bernoulli sampling, paired HT estimates.

`r2w/gates.py` — support gate, hard number/date/entity checks, HQ answerability.

`r2w/costs.py` — four resource ledgers and net-value equations.

`r2w/model.py` — double-contrast LightGBM/GBM ensemble interfaces.

`r2w/policy.py` — stage-1 entry gate and stage-2 raw-protected safe set.

`r2w/offline.py` — raw/mixed/policy-background replay orchestration with one candidate bundle per memory.

`r2w/online.py` — deployment flow and atomic writer interface.

`r2w/folds.py` — fixed LoCoMo outer folds.

`r2w/artifacts.py` — versioned label rows, sampling audit and separate resource components.

`r2w/data.py` — strict LoCoMo and LongMemEval adapters into the v4 replay schema.

## Formal experiment adapters

The package deliberately depends on explicit frozen runtime adapters rather than silently switching models:

- `StructuredConstructor.json(prompt)`
- `Embedder.encode(...)`
- `SupportModel.score(claim, source)`
- `QAChecker.answerable(question, source)`
- `QueryRunner.score(query, bodies)` with fixed `reader_version/scorer_version/prompt_version/seed`
- `ResourceMeasurer` for actual select/commit/index/read accounting
- `ParentWriter.atomic_write(...)`

This prevents a failed external API from silently changing the builder/reader/support model inside one label version. A different provider/model version is a different experiment snapshot and requires replay.

There is no bundled provider setup script or zero-cost runtime adapter. Tests use local fakes; production must inject measured adapters explicitly.

`rows_from_replay` also rejects a label artifact unless it declares frozen versions for the constructor, retriever, reader, scorer, prompt, tokenizer, support model, QA checker and resource-pricing policy.

## Privacy boundary

Source code and test fixtures use only generic identifiers and synthetic values. `r2w/data.py` reads benchmark records only from a path supplied at runtime; it does not embed benchmark conversations, names, locations, account identifiers, or contact data in the package.

## Validation

Run the current algorithm regression suite:

```bash
./scripts/run_current_checks.sh
```

or:

```bash
python -m unittest discover -s tests -t . -v
python -m compileall -q r2w tests
```

The tests cover: 6-action core, extended opt-in, one candidate call, reject-not-truncate contracts, parent-level top-k, paired HT and double-contrast identity, stage-1 rejection, stage-2 `none`, raw protection, four resource groups, natural LoCoMo folds, and full online selection with fake frozen adapters.
