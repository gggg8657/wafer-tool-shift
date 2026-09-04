# critique log — wafer-tool-shift

Running, adversarial notes on this repo's own results. Every number here comes
from a run in this repository or from a measurement script named at the point
the number appears. Published figures, where they appear at all, are labelled as
published and attributed.

Sessions are appended, newest last.

---

## 2026-09-04, session S1 (unattended)

State at the start: 59 measured cells in `runs/`, `RESULTS.md` at commit
`373b9cc`, GPUs 0 and 1 idle, nothing running.

### 0. First, a claim I was told to commit and found already committed

The brief said "`RESULTS.md` was regenerated after two reporter bugfixes but not
committed. Commit and push it." It was already committed and pushed: `HEAD` and
`origin/main` were both `373b9cc`, and regenerating from a copy of the Friday
`runs/` reproduced the committed file byte for byte:

```
$ python scripts/report.py --runs runs.bak_0904_1059 --out /tmp/RESULTS_check.md
wrote /tmp/RESULTS_check.md from 59 cells
$ diff <(git show HEAD:RESULTS.md) /tmp/RESULTS_check.md   # no output
```

Recording this because "the task list said so" is not evidence, and the
reproduction is worth more than the commit: it establishes that `RESULTS.md` is
a pure function of `runs/`, which is the property the rest of this log leans on.

### 1. The best cell's fourth channel: the claim was not measurable, and now it is

**The claim under test.** `rpca_cnn` is the best `lot` cell at macro-F1 0.8813
against 0.8671 for the same CNN without the fourth channel (`cnn_gn`), and the
stated mechanism is that per-lot Robust PCA splits each lot's wafers into a
low-rank tool signature `L` and a sparse per-wafer defect `S`, and the encoder
is handed `S` — the defect with the lot's nuisance removed.

**What the decomposition actually returns.** Measured directly from
`data/rpca.pt` and `data/corpus.pt` in this session:

| quantity | value |
|---|---|
| wafers with a decomposition at all (lot has >= 12 wafers) | 164,824 / 172,948 |
| of those, fraction with `rank(L) == 0` | **0.9483** |
| rank histogram over decomposed wafers | 0: 156,303, 1: 8,391, 2: 105, 3: 25 |
| fraction of *all* wafers where residual is bit-identical to the raw fail mask | **0.9527** |
| mean per-wafer L1 distance residual vs fail mask, rank-0 wafers | 4.9e-09 |
| same, on the 8,521 wafers with rank >= 1 | 311.6 (58% of the wafer's failed-die mass) |

So for 19 wafers in 20 the "lot signature channel" is a bitwise copy of the
failed-die indicator, which the 3-channel one-hot input already carries as
channel 2. The mechanism story is false on 95% of the corpus by construction,
before any model is trained. The decomposition is real and does substantial work
on the 4.9% of wafers where a lot genuinely shares a component — that part is
not in dispute — but it cannot be what moves a corpus-level macro-F1.

**Why this was invisible.** The signature column that records the rank is
written into `data/rpca.pt` (`signature[:, 4]`) and never surfaced in any table.
Nothing in `RESULTS.md` reported a property of the decomposition itself; it
reported only the downstream accuracy of a cell whose name contains "rpca".

**The distinguishing experiment.** Three explanations predict different things,
so replace the fourth channel and hold everything else fixed
(`scripts/ablate_sigchannel.sh`, `--sig-channel`):

| fourth channel | if this wins, the explanation is |
|---|---|
| `residual` (RPCA sparse part) | the decomposition is doing the work — the claim |
| `failmask` (raw failed-die mask) | a redundant copy of an existing channel is enough |
| `zeros` | nothing is in the channel; the extra conv filters / init are the effect |

**Result so far** (protocol `lot`, 12 epochs, seeds as listed; the sweep is
still filling in `lot_time` and `size`):

| fourth channel | seeds | per-seed test macro-F1 | mean |
|---|---|---|---|
| none (`cnn_gn`, 3ch) | 0,1,2 | 0.8671, 0.8680, 0.8591 | 0.8647 |
| `residual` (the claim) | 0,1,2 | **0.8813**, 0.8654, 0.8621 | 0.8696 |
| `failmask` (control) | 0 | 0.8765 | — |
| `zeros` (control) | 0 | 0.8772 | — |

The headline 0.8813 is the top of its own seed range, not a central estimate:
the same cell gives 0.8621 at seed 2. The `residual` spread on `lot` is
0.8621–0.8813, a range of 0.0192, and the claimed effect was +0.0142. A channel
of **zeros** scores 0.8772 at seed 0, above the `residual` three-seed mean.

**The conclusion I will defend.** The RPCA lot-signature channel is not the
reason `rpca_cnn` was the best cell. On present evidence the ranking of
`rpca_cnn` above `cnn_gn` in `RESULTS.md` is a single-seed artefact, and
whatever small residual advantage a fourth channel has is available from a
channel containing no information. `README.md` and `RESULTS.md` must say this
where the number appears.

**What would change my mind**, and it is worth stating because I have only three
seeds: if `failmask` and `zeros` come in materially below `residual` once all
three seeds and all three protocols are in, the decomposition is contributing
something on the 4.9% minority and the effect is simply small. The seed-spread
table is the arbiter, and it is regenerated by `scripts/report.py`, not typed.

**A methodological hole this exposes, larger than the cell itself.** Nobody had
measured the seed spread on this corpus, and `RESULTS.md` reported 59 cells at
one seed each. The measured `lot` spread for a single configuration is ~0.019
peak-to-peak. Read across the existing tables, that is larger than *every*
domain-generalization delta previously called "within noise (±0.005)" — the
verdict was right, but the ±0.005 was an assertion, not a measurement — and it
is larger than the entire spread between the top four representations. Two
consequences, both of which I am acting on:

1. `report.py` now records the seed on the key. It previously keyed cells by
   `(protocol, encoder, objective, tag)`, so a second seed of any cell would
   have silently overwritten the first and whichever file sorted last would have
   won. Nobody would have seen a warning.
2. Every existing table is now labelled seed-0-only in the document itself.

**One caveat on the seed spread, so it is not over-read.** `seed` reshuffles
three things at once: model init, which training domains become the inner
validation split (`Runner._inner_split`, `seed + 1`), and — for `lot` and
`size` — which groups land in the test set (`_stratified_group_split`). For
`lot_time` the test set is fixed regardless of seed, because
`split()` ignores `seed` on that branch (`wts/data.py:209-228`). So `lot_time`
seed spread is init-plus-validation only and is expected to be the smaller of
the two. That asymmetry is a property of the protocols, not a bug, but it means
the two spreads are not the same quantity.

### 2. `sinkhorn` collapsed to the class prior, and the loss curve says why

Measured, from `runs/lot__cnn_bn__sinkhorn__s0.json` and
`runs/lot__feat__sinkhorn__s0.json`:

- test macro-F1 0.1026 (`cnn_bn`) and 0.3331 (`feat`), against 0.8587 and 0.8417
  for the same encoders under ERM on the same protocol.
- `cnn_bn` val macro-F1 is 0.1018 at **every** epoch from 1 to 12.
- train loss: 1.1607, 0.8727, 0.8557, 0.7350, 0.6784, 0.6666, 0.6643, 0.6630,
  0.6622, 0.6618, 0.6616, 0.6616 — monotone, converged, flat at 0.6616.
- `feat` reaches val 0.3229 at epoch 1, then drops to 0.1018 at epoch 2 and
  stays there; its loss goes *up* at epoch 2 (1.0225 -> 1.2797) before settling
  at 0.664.

This is not divergence and it is not a NaN — the optimizer is converging
smoothly to a fixed point where the classifier emits the training class prior.
The obvious alternative explanation, "it needs more epochs", is refuted by the
loss being flat to four decimals for the last four epochs and by the val metric
never moving off 0.1018 from epoch 1.

The mechanism I claim: `sinkhorn_divergence` (`wts/methods.py:223-242`) is
minimized exactly by making every domain's embedding cloud identical, and the
cheapest way to do that is to collapse the embedding to a point. With
`--ot-lambda 1.0` the penalty is on the same scale as the classification loss it
is competing with, so collapse is the better trade. Note the normalization
`C = C / C.detach().max()` makes the penalty scale-invariant, so it cannot be
gamed by shrinking the embedding — only by genuine collapse, which is what the
flat prior-predicting output looks like.

**Distinguishing experiment**, running: `scripts/sinkhorn_lambda.sh` sweeps
`--ot-lambda` over 0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0 on `lot`/`cnn_bn`, with
the per-epoch OT penalty now logged into the run JSON (`run_bench.py` was
recording the objectives' aux dict as `_` and discarding it). Two outcomes:

- if macro-F1 recovers at some lambda where the logged OT penalty is still
  meaningfully below its lambda=0 value, the implementation is fine and 1.0 was
  simply far too large — a tuning result, and the row stays with its weight
  reported;
- if macro-F1 only recovers where the penalty is effectively unpenalized, the
  objective does nothing useful here and **the row comes out of the table**
  rather than sitting there as a broken cell that looks like a finding.

lambda = 0 is included as an ERM control run through the *same* objective
wrapper, so the comparison is not confounded by the wrapper itself.

### 3. Why the SSL-initialized cells never produced a result: a shell bug, not a model failure

`runs/` had no `sslinit` file although `runs/ssl_pretrain.pt` existed and
`logs/sweep_c.log` says all three cells were launched (09:47:18, 09:49:20 x2).
There is no `logs/c_*sslinit*.log` either, which is the tell.

Root cause, reproduced:

```
$ spec="--encoder cnn_gn --objective erm --protocol lot --init-from runs/ssl_pretrain.pt --tag sslinit"
$ echo "$spec" | tr -d ' -' | tr '[:upper:]' '[:lower:]' | cut -c1-70
encodercnn_gnobjectiveermprotocollotinitfromruns/ssl_pretrain.pttagssl
```

`tr -d ' -'` strips spaces and dashes but not the slash in the checkpoint path,
so the redirect target was
`logs/c_...runs/ssl_pretrain.pttagssl.log` — a directory that does not exist.
bash fails to open the redirect *before* exec, so the python process never
started, `$!` still produced a PID for the failed subshell, `wait` returned, and
the sweep logged "launch" and moved on. Three cells reported as run, never run,
no error anywhere a person would look.

Fixed in `scripts/sweep_c.sh` and in all three new scripts with
`slug(){ echo "$*" | tr -cs 'A-Za-z0-9' '_' | ...; }`, which cannot emit a path
separator, plus an explicit post-launch check that the log file exists.

This is the second reporting-layer bug in two hours (the other is the
seed-collision key in `report.py`). Both share a shape: a silent overwrite or a
silent drop, with no failure surface. That is worth more attention than any
single model result, because both of them destroy evidence rather than produce a
wrong number, and a wrong number at least gets argued with.

**The honest prior for what the fixed cells will show**, written down before the
run so it can be scored: `logs/ssl.log` reports the adversarial nuisance CE over
8 epochs as 5.0889, 4.9570, 4.8939, 4.8877, 4.9902, 5.2285, 4.9598, 5.0374
against a chance value of 5.7683. The adversary is supposed to push this *up*
toward chance; it ends 0.73 nats below chance and is not monotone. So the
embedding demonstrably still carries the nuisance, and I expect the sslinit
cells to land inside the seed spread of `cnn_gn` from scratch. If that is what
happens, it is a negative result about lot-adversarial pretraining on 638,506
unlabelled wafers, reported as one — not a reason to tune until it wins.

### 4. Second opinion: `codex` found something worse than anything above

Prompt given: *"Read `scripts/run_bench.py`, `wts/methods.py` and `wts/data.py`.
Find the strongest specific reason the reported macro-F1 numbers might be
inflated, leaked, or not comparable across cells. Be concrete about function
names and line numbers. Rank your top 3."*

`cursor-agent` returned `Error: Authentication required` and produced nothing;
it cannot be used in this unattended session. `agy` not yet tried.

#### codex #2 — upheld, and it invalidates the headline negative result

> **"Domains" are collapsed modulo 32, so domain-aware objectives do not operate
> on actual lots or geometries.** `batch_of()` replaces the true domain with
> `self.dom[sel] % N_BUCKETS` [...] This aliases roughly 10,762 lots — or 346
> sizes — into only 32 arbitrary buckets. `group_dro`, `dann`, `irm`, and
> `coral` then treat those collisions as genuine domains. Consequently,
> comparisons advertised as lot- or size-aware are really comparisons using
> different amounts and patterns of hash collision.

This is correct and I had not seen it. Measured here, mean pairwise
total-variation distance between the class distributions of the groups:

| domain vocabulary | groups (>=200 wafers) | mean pairwise label TV |
|---|---|---|
| `lot % 32` — **what every objective actually saw on `lot`** | 32 | **0.0208** |
| real lots | 6,504 | 0.1666 |
| lot production-order decile | 10 | 0.1822 |
| lot failed-die-rate decile | 10 | 0.1231 |
| real geometries | 58 | 0.4466 |
| `size_id % 32` — what the objectives saw on `size` | 31 | 0.2592 |

With ~336 lots averaged into each of 32 buckets, every bucket converges on the
corpus marginal. So on the `lot` protocol the invariance objectives were asked
to equalize 32 distributions that are already equal to within TV 0.02 — a
condition each of them satisfies by doing nothing. **"Every borrowed
domain-generalization objective failed to beat ERM on `lot`, deltas within
±0.005" is therefore not evidence about the objectives.** It is close to a
tautology given the domain definition, and the fact that the deltas came out
suspiciously *small* — smaller than the seed spread measured in §1 — is exactly
what that tautology predicts.

Note what this does **not** touch, and I want this stated because it would be
convenient to let the whole negative result collapse: on the `size` protocol the
same hash is far less degenerate (344 geometries into 32 buckets, TV 0.2592),
and the `size` column is where the objectives showed large, real, mostly
*negative* effects (GroupDRO −0.18 to −0.27, logit adjustment −0.08 to −0.10).
Those cells were operating on domains that genuinely differ. The `size` half of
the negative result stands; the `lot` half does not.

**The fix and the hypothesis, written before the run.** `run_bench.py` now
separates two things that were conflated in one `self.dom`: the protocol's
grouping (what is held out, what worst-domain metrics are computed over) and the
invariance domain handed to the objectives. `--domain-def` selects the latter;
`hash32` is the default and is bit-identical to the old expression, verified by
a test, so no existing cell changes meaning.

> **H4:** the ERM-equivalence of GroupDRO / IRM / CORAL / DANN / HSIC /
> domain-mixup on the `lot` protocol is an artefact of the domain definition.
> Under `time_decile` — ten fixed, ordered, lot-level production-order groups
> with label TV 0.1822, the axis this repo has already shown carries the largest
> real shift — at least some of these objectives should move measurably off ERM,
> in either direction. If all six still sit inside the seed spread from §1, the
> negative result survives a far stronger test and becomes a much better claim
> than it was.

`scripts/domain_def_sweep.sh` runs `cnn_bn` x {erm, group_dro, irm, coral, dann,
mixup_domain, hsic} x {hash32, time_decile} x 3 seeds on `lot`, queued behind
the current jobs. `erm` is included under both definitions as a null control: it
never reads the domain, so the two must agree exactly at a given seed, and if
they do not the plumbing changed something and nothing else in the sweep can be
read.

#### codex #1 — real, but it is not leakage, and I am not fixing it blind

> **Test-label leakage in domain selection.** `_stratified_group_split()`
> examines the labels of each candidate held-out lot and rejects domains whose
> removal would hurt training-class coverage [...] Thus target labels influence
> which target domains become the test set.

The mechanism is correctly described (`wts/data.py:161-167`); the word
"leakage" is wrong. No test label reaches the model, the loss, or model
selection — the split is label-*aware*, which biases which domains land on the
test side, and that is a different and smaller problem than leakage. The guard
fires only when moving a group would drop a class below one training example,
so it binds almost exclusively on `Near-full` (149 wafers in the whole corpus)
and it pushes rare-class lots into *training*, which if anything makes the test
side easier on the rare classes and the macro-F1 optimistic. So the direction
codex asserts is plausible. The magnitude is unmeasured and I will not assert
one. Logged as a task: run `lot` with a label-blind group split and report both.

#### codex #3 — real, latent, no cell affected, now guarded

> **Spectral cells silently disable or change the domain-aware objectives.**
> `size_bucketed_batches()` makes every spectral batch contain exactly one
> geometry [...] Under the `size` protocol, geometry is the domain, so each
> batch has one domain. Therefore CORAL's penalty is always zero [...]

Correct reasoning. Checked against `runs/`: the only spectral cells that exist
are `{iid, lot, size, lot_time} x erm`, so no reported number is affected — the
comparability failure codex describes is latent, not realized. Under the `lot`
protocol a spectral batch spans one geometry but many lots, so domains do vary
within it; only `size` is degenerate. `run_bench.py` now refuses that
combination with an error naming the reason rather than producing an
ERM-in-disguise cell.

Score for the exercise: one finding that changes a headline claim, one that is
real but mis-named and needs measuring, one latent bug worth a guard. Cheap.
