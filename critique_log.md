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

### 5. The fourth-channel ablation, completed: the RPCA claim does not survive

30 cells, 12 epochs, 3 seeds each, `scripts/ablate_sigchannel.sh`, regenerated
into `RESULTS.md` by `report.py`:

| protocol | fourth channel | seeds | per-seed test macro-F1 | mean | half-range | vs `cnn_gn` |
|---|---|---|---|---|---|---|
| `lot` | none (`cnn_gn`) | 3 | 0.8671, 0.8680, 0.8591 | 0.8647 | ±0.0044 | — |
| `lot` | RPCA residual | 3 | 0.8813, 0.8654, 0.8621 | 0.8696 | ±0.0096 | +0.0049 |
| `lot` | raw fail mask | 3 | 0.8765, 0.8678, 0.8634 | 0.8692 | ±0.0065 | +0.0045 |
| `lot` | **zeros** | 3 | 0.8772, 0.8673, 0.8622 | 0.8689 | ±0.0075 | +0.0042 |
| `lot_time` | none | 3 | 0.6935, 0.6993, 0.7026 | 0.6985 | ±0.0045 | — |
| `lot_time` | RPCA residual | 3 | 0.7020, 0.7159, 0.7084 | 0.7088 | ±0.0070 | +0.0103 |
| `lot_time` | raw fail mask | 3 | 0.6905, 0.7008, 0.7098 | 0.7004 | ±0.0096 | +0.0019 |
| `lot_time` | zeros | 3 | 0.6991, 0.7011, 0.7051 | 0.7018 | ±0.0030 | +0.0033 |
| `size` | none | 3 | 0.8203, 0.8301, 0.8895 | 0.8467 | ±0.0346 | — |
| `size` | RPCA residual | 3 | 0.8254, 0.8146, 0.8863 | 0.8421 | ±0.0359 | −0.0046 |
| `size` | raw fail mask | 3 | 0.8287, 0.8145, 0.8832 | 0.8421 | ±0.0343 | −0.0045 |
| `size` | zeros | 3 | 0.8236, 0.8167, 0.8790 | 0.8398 | ±0.0312 | −0.0069 |

**Verdict on `lot`, which is where the claim was made.** The three fourth-channel
variants land at +0.0049, +0.0045 and +0.0042 over the 3-channel baseline. They
are separated from each other by 0.0007 — two orders of magnitude below the
effect anyone would report — and every one of them is inside its own seed
half-range of the baseline. **A channel containing nothing but zeros buys as
much as the RPCA decomposition does.** Whatever the fourth channel is worth on
this protocol, it is worth it as extra first-layer capacity and a different
initialization, not as information about the lot's tool signature. The
prediction from §1 (rank 0 on 94.8% of lots, residual bit-identical to the fail
mask on 95.3% of wafers) is confirmed downstream.

The published 0.8813 is the maximum of a three-seed range whose minimum is
0.8621. Reporting it as "the best cell on `lot`, 0.8813" against "cnn_gn 0.8671"
compared the top of one range against the middle of another.

**Where I will not overclaim.** On `lot_time` the residual is nominally the
highest of the three (+0.0103 vs +0.0019 and +0.0033) and it beats both controls
at every seed. That is the one place the decomposition might be doing something,
and the forward-only protocol is exactly where a genuine tool signature *should*
matter most. But the separation is 0.007–0.008 against a control half-range of
±0.0096, with n=3. It is a hypothesis worth another 5 seeds, not a result.
Recorded as such and not written into any headline.

**A second, larger finding that fell out of running seeds at all.** The `size`
protocol has a seed half-range of **±0.031 to ±0.036** — seed 2 scores
0.879–0.890 for all four variants while seeds 0 and 1 score 0.815–0.830. That is
because `_stratified_group_split` reshuffles which *geometries* are held out,
and geometries are wildly heterogeneous, so `size` seed spread is a property of
the protocol rather than of any model. Consequences for what is already
published:

- Every `size`-protocol delta smaller than ~0.03 in `RESULTS.md` is noise. That
  covers most of the objective comparisons on that protocol.
- It does **not** rescue the large negatives: GroupDRO at −0.18 to −0.27 and
  logit adjustment at −0.08 to −0.10 are five to eight times the seed
  half-range, so those remain real effects.
- `lot` (±0.004–0.010) and `lot_time` (±0.003–0.010) are far tighter, and
  `lot_time` should be tighter still since its test set does not move with seed.

**What this costs, stated plainly.** The claim "best cell on `lot`: CNN with an
RPCA lot-signature channel, 0.8813" is withdrawn. The corrected statement is
that on `lot`, `cnn_gn` with any fourth channel and `cnn_gn` without one are
indistinguishable at three seeds, mean 0.865–0.870. This is a negative result
about a method this repo invented for itself, which makes it the least
convenient kind and the one most worth keeping.

### 6. The SSL-initialized cells, once they actually ran: pretraining costs 5–8 points

With the redirect bug fixed the nine cells ran. `--init-from` reports
`initialized 26/26 tensors`, i.e. the whole encoder, so the transfer is total,
not partial.

| protocol | from scratch | SSL-initialized | difference |
|---|---|---|---|
| `lot` | 0.8647 ±0.0044 — 0.8671, 0.8680, 0.8591 | 0.8143 ±0.0091 — 0.8134, 0.8239, 0.8056 | **−0.0504** |
| `size` | 0.8467 ±0.0346 — 0.8203, 0.8301, 0.8895 | 0.7711 ±0.0265 — 0.7602, 0.7500, 0.8030 | **−0.0756** |
| `lot_time` | 0.6985 ±0.0045 — 0.6935, 0.6993, 0.7026 | 0.6345 ±0.0225 — 0.6120, 0.6569, 0.6345 | **−0.0640** |

On all three protocols the two seed ranges do not overlap at all: the best
SSL-initialized seed is below the worst from-scratch seed every time. At n=3
that is about as clean as this design can give. Lot-adversarial masked-die
pretraining on 638,506 unlabelled wafers does not help under shift — it costs
five to eight points.

This is consistent with the pretraining log rather than a surprise on top of it.
The adversary was supposed to push the nuisance cross-entropy up toward its
chance value of 5.7683; over 8 epochs it went 5.0889, 4.9570, 4.8939, 4.8877,
4.9902, 5.2285, 4.9598, 5.0374 — traction, never arrival, not monotone. An
embedding that still predicts the nuisance 0.73 nats better than chance has not
been made invariant to it; it has been made *worse at the thing we then ask it
to do*, since the reconstruction objective and the reversal are both pulling the
representation away from class-discriminative structure.

**The confound I have to rule out before this goes in a paper.** Every cell
above fine-tunes at `lr 2e-3` on a OneCycle schedule, and that LR was chosen for
a random initialization. It is not obviously right for pretrained weights, whose
scale is different: measured from the checkpoint, the deepest conv has mean |w|
0.0852 against 0.0147 for a freshly constructed encoder, a factor of 5.8. A
schedule peaking at 2e-3 may simply be destroying what it was given, in which
case the honest claim is "this pretraining does not survive this recipe", not
"this pretraining is bad".

Two things I checked before assuming the confound was elsewhere:

- The classifier head is **not** a stale head from another task. The checkpoint
  carries `head.weight` of shape (9, 128) because `MaskedDieModel` wraps a
  whole `CnnResized`, but pretraining only ever calls `enc.embed()`, so that
  head never received a gradient. Its statistics match a fresh init to three
  decimals (mean |w| 0.04374 vs 0.04419, std 0.05057 vs 0.05104). Loading it is
  equivalent to a different random draw. So no `--init-body-only` variant is
  needed and I did not add one.
- `initialized 26/26` confirms nothing was silently skipped on a shape mismatch,
  which would have made this a partial-transfer result masquerading as a full one.

`scripts/ssl_lr_sweep.sh` runs **both arms** at 2e-4, 5e-4, 1e-3 (2e-3 is
already measured for both), two seeds each, and compares LR by LR. Sweeping only
the pretrained arm and putting its best against the single from-scratch cell
would be selection on the treatment — the mirror image of the mistake that made
`rpca_cnn` look like a winner.

**Prediction, recorded before the sweep runs:** if the deficit is a recipe
mismatch, the sslinit curve should be much flatter in LR than the from-scratch
curve and should close most of the gap at 2e-4. If sslinit is below scratch at
every LR, the representation is genuinely worse and the negative result stands
as stated.

### 7. Second opinion: `agy` on the documents rather than the code

Prompt: *"Every claim in `RESULTS.md` and `README.md` must be backed by a run in
`runs/*.json`. Find claims that are overstated, unsupported by the JSON, or that
a peer reviewer would reject. Quote the exact sentence. Do not praise
anything."* Six findings; three change something.

#### upheld, and the best single catch of the session — the p10 column is quantized

> *"In `RESULTS.md` (protocol `lot`), 16 distinct models/objectives report an
> identical p10 domain F1 of 0.4898 [...] a 25-wafer lot containing 24 `none`
> wafers and 1 defect wafer yields an exact macro-F1 of (48/49 + 0)/2 =
> 0.4897959 whenever the single defect is missed. It is a discretization
> artifact of lot size, not a discriminative measure of domain robustness."*

Verified, and it is worse than agy said. Counting `p10_domain_macro_f1` over
every `lot` cell in `runs/`:

```
  0.489796  x25        <- exactly 24/49
  0.500000  x11
  0.318841  x1
  0.234043  x1
  0.541063  x1
```

25 cells share one value and 36 of 39 share one of two values. `README.md` says
*"A lot holds at most 25 wafers, so the single worst lot is noisy and the p10 is
the headline"* — the reasoning is right about why the worst lot is noisy and
wrong to conclude the p10 fixes it. Both statistics are quantized by the same
cause, and no size floor rescues it, because no lot has enough wafers: the
quantization is a property of the domain size, not of the threshold. My first
attempt at a fix was `min_n=64`, which would have excluded every lot and
returned an empty statistic — caught before it shipped.

The fix that works is a statistic that aggregates *across* domains so it moves
continuously even though each domain's score does not: `mean_domain_macro_f1`
and `frac_domains_below_half`, both means over ~1,700 lots. Added to
`wts.metrics.summarize`; cells measured before the change do not carry them and
`RESULTS.md` says so rather than leaving blanks to be misread.

This one stings because the p10 column was presented as the honest headline —
the metric that was supposed to stop a model hiding a bad tool behind a good
average — and it could not distinguish any two models on the protocol it
mattered most for.

#### upheld — ECE and conformal coverage are both marginal on an 85%-`none` corpus

> *"Claiming ECE cannot be gamed under extreme class imbalance is false."*

Correct. `ece()` weights each confidence bin by its share of samples, `none` is
~85% of the corpus and is predicted confidently and correctly, so the statistic
sits below 0.01 across every model regardless of what macro-F1 does. `README.md`
lists it under *"metrics that a model cannot game"*, which is an overstatement.
Annotated in `RESULTS.md`.

> *"`hit.mean()` is a single scalar representing overall marginal coverage [...]
> Class-conditional coverage is never stored in `runs/*.json`."*

Half right, and the half that is right is the half that matters. The
*construction* is class-conditional — `classwise_conformal` computes a separate
quantile per class, exactly as `README.md` claims — so "never class-conditional"
is wrong. But averaging the hits over the test set collapses it back to a
marginal number that `none` dominates, which is the failure the class-conditional
construction exists to prevent. The claim is defensible about the method and
misleading about the reported number. `summarize` now emits
`conformal_coverage_per_class`, `conformal_coverage_worst_class` and
`conformal_empty_set_rate`; agy's observation that mean set size below 1.0
implies empty prediction sets is what prompted the last of those.

#### upheld, already in flight — the collapsed `sinkhorn` row

> *"Presenting a collapsed run from an uncalibrated hyperparameter alongside
> tuned baselines as evidence that optimal transport fails on this benchmark
> would be dismissed as evaluating a strawman."*

Agreed, and it is the same conclusion §2 reached independently.
`scripts/sinkhorn_lambda.sh` is running. The row stays visible with its
diagnosis attached rather than being quietly filtered, because a table that
drops its embarrassments is worse than one that explains them.

#### upheld — my own sentence about model selection was self-contradictory

> *"Validation selection and test selection chose the exact same configuration
> with the exact same test score. Labeling 0.8813 a 'selection artefact' under
> test selection while asserting it is valid under validation selection is
> contradictory."*

Correct, and it is a sentence I wrote two hours earlier in this session. When the
two selection rules coincide there is nothing to warn about, and printing the
warning anyway reads as though a distinction was found where none was. To be
fixed in `report.py`: say the two rules agreed, and only contrast them when they
disagree.

#### rejected — no test leakage in the active-learning loop

> *"`scripts/active_learning.py:130` evaluates candidate models on `te[:2048]`
> (test split data) during the active acquisition loop."*

The line is real and the inference is wrong. `train_eval` trains for a fixed
epoch count with no early stopping and no model selection, then returns
`(model, predictions)`; the acquisition loop calls it as `model_now, _ =
train_eval(..., te[:2048], ...)` and **discards the predictions**. Nothing
derived from the test split reaches the acquisition scores, which come from
`lot_scores` on the *pool*. The cost is 2,048 wasted forward passes per
acquisition step, not a leak. Worth deleting for clarity; not worth a
correction to any number.

#### rejected on a detail — `min_per_class` is 1, not 5

agy quotes `min_per_class` as 5 in `_stratified_group_split`; the default in the
signature is 1. The substantive point about label-aware test-split construction
is the same one codex raised and is logged in §4 as still needing a measurement.

#### upheld as a wording problem — TTA is described as deployable, and it hurts

> *"Framing them as viable production tools without acknowledging that they
> degrade performance across shifted distributions is misleading."*

`README.md` calls AdaBN/TENT *"what a fab could actually deploy"*, and the
measured effect is negative on every shifted protocol. The sentence describes
the *setting* correctly — these methods need only unlabelled target wafers —
but reads as an endorsement. The finding that the deployable methods make things
worse is one of this repo's better results and the README should lead with it
rather than bury it.
