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

### 8. The forward-only protocol is not purely temporal, and that is a confound in the largest result

`lot_time` is the headline: macro-F1 falls from ~0.86 lot-disjoint to 0.64–0.70
forward-only. The whole thing rests on `wts.data.lot_numbers`, which reads the
integer out of a lot name and treats it as a clock. Nobody had tested that.

**Test 1 — is the numbering arbitrary?** It cannot be proved to be time, but it
can be refuted as random. Bucket lots into deciles of lot number, take the
total-variation distance between each pair of deciles' distributions over a
wafer-level variable, and correlate that against the decile gap. The null is 200
random reassignments of numbers to lots, which preserves every lot's contents
and every marginal and destroys only the ordering (`scripts/time_proxy_check.py`
→ `runs/time_proxy.json`):

| variable | Spearman(gap, TV) | null p95 | p | TV adjacent | TV farthest |
|---|---|---|---|---|---|
| geometry | +0.3834 | 0.2800 | 0.000 | 0.8682 | 0.9998 |
| defect class | +0.2560 | 0.3005 | **0.055** | 0.0964 | 0.4129 |
| failed-die rate | +0.4050 | 0.2452 | 0.000 | 0.4062 | 0.7651 |

Geometry and process health drift monotonically with lot number; the defect
class mix does not, at p = 0.055 and with its correlation *below* its own null's
95th percentile. So the numbering carries real systematic structure — the "these
are arbitrary IDs" objection is dead — but the test cannot distinguish
production order from product blocking, because a product line numbered in a
contiguous block produces exactly this signature. That is stated in the paper
rather than glossed into "we verified the time axis".

**Test 2 — and this is the one that hurts.** The adjacent-decile geometry TV is
**0.8682**. Neighbouring tenths of the lot numbering share almost no geometry at
all. That is not gentle drift; it says the numbering is heavily blocked by
product. Which raises the obvious question nobody had asked: what geometries does
the forward-only *split* actually test on? Measured from the splits alone, no
model involved:

| protocol | geometries in train | in test | unseen in train | test wafers of unseen geometry |
|---|---|---|---|---|
| `iid` | 327 | 266 | 17 | 0.05% |
| `lot` | 320 | 209 | 24 | 0.28% |
| `size` | 205 | 139 | 139 | 100.00% |
| **`lot_time`** | **338** | **19** | **4** | **14.15%** |

The forward-only test set is **19 geometries**. Training sees 338. One test wafer
in seven is of a geometry the model has never seen, against one in 360 for the
plain lot holdout. So `lot_time` is not a clean temporal protocol: it is a
forward-only split that also concentrates the test distribution onto a narrow
slice of geometry space and puts a seventh of it out of distribution
geometrically. Part of the ~0.17 drop credited to "forward-only time" is the
`size` protocol arriving through the back door.

**How large a part is currently unmeasured, and I am not going to guess.**
`run_bench.py` now decomposes every test set into its seen-geometry and
unseen-geometry halves and evaluates both, so the next run of any cell answers
it. Queued as the first stage of `chain_rest.sh`, promoted ahead of three other
sweeps precisely because it bears on the biggest claim.

**A trap in that decomposition that I built the guard for up front.** macro-F1
averages over the classes *present*, and the seen and unseen halves do not
contain the same classes — the unseen half is 19 geometries' worth of wafers and
will be missing several. Comparing their macro-F1 values directly would be
comparing averages over different class sets, which is the same error as
comparing across protocols without saying so. Both halves therefore record their
full per-class F1 and their class list, so a like-for-like average over shared
classes can be taken downstream, and the paper prints the caution next to the
columns rather than in a footnote.

**What this does not do.** It does not withdraw the forward-only result. The
protocol is still the honest one for deployment — a fab genuinely does meet new
products and new tools going forward, and a split that forbids that is the
optimistic one. What changes is the *explanation*: "forward-only time costs 20
points" becomes "forward-only deployment costs 20 points, of which an
unquantified share is meeting geometries you have not trained on". The second
sentence is less quotable and more nearly true.

### 9. The sinkhorn sweep, partway: the control is doing its job and has already found something

Two of seven cells (`scripts/sinkhorn_lambda.sh`), with the per-epoch OT penalty
now logged into the run JSON:

| `--ot-lambda` | test macro-F1 | OT penalty, epoch 1 → 12 | Scratch F1 |
|---|---|---|---|
| 0.0 (ERM through the same wrapper) | 0.8609 | 0.1360 → 0.1691 | 0.7224 |
| 0.003 | 0.8591 | 0.1360 → 0.1699 | 0.6992 |
| 1.0 (the original cell) | 0.1026 | not logged | 0.0000 |

Two things worth recording before the rest lands.

First, the penalty at λ = 0.003 ends at 0.1699 against 0.1691 unpenalized — the
penalty is not being reduced at all, so that weight is doing nothing whatsoever.
The unpenalized penalty *rises* over training, which is what an embedding that
is getting more discriminative and therefore more domain-separated should do.
Whatever the sweep finds, the useful window is somewhere above 0.003 and below
1.0, and the two ends bracket it.

Second, and this was not what the sweep was for: **λ = 0 scores 0.8609 where the
stored plain-ERM cell for the same encoder, protocol and seed scores 0.8587.**
Those two are mathematically the same computation — with λ = 0 the penalty term
contributes exactly zero to the gradient — differing only in that `sinkhorn`
routes through `model.embed()` then `model.head()` while `erm` calls `model(x)`,
and in that the penalty is computed and discarded. A gap of 0.0022 between two
configurations that are algebraically identical is a *floor on same-seed
reproducibility*, and it is larger than several deltas this repo has reported as
findings.

I have not established whether that 0.0022 is non-deterministic convolution
backward kernels, or a real difference in floating-point op ordering between the
two code paths, and I will not assert either. What I did instead is make the
backfill measure it: `scripts/backfill_metrics.sh` now runs one cell **twice
under identical arguments** before overwriting anything, treats the difference
between those two runs as the run-to-run floor, and admits the backfill only if
the stored values agree with a fresh re-run to within it. The previous version
demanded agreement to 1e-9, i.e. assumed bit-reproducibility on a GPU, which
nobody had tested and which would have aborted the whole stage on a property
that is not a bug. The floor goes to `runs/determinism.json` for the reports to
cite, because it bounds every same-seed comparison here.

### 10. The sinkhorn verdict: not a strawman, and not a tuning failure

Six of seven weights in (λ = 1.0 still running with logging; the original
untagged λ = 1.0 cell already sits at 0.1026). Selection on the domain-disjoint
validation split, never on test:

| `--ot-lambda` | val macro-F1 | test macro-F1 | OT penalty ep1 → ep12 | penalty vs unpenalized |
|---|---|---|---|---|
| 0.0 (ERM through the wrapper) | 0.8796 | 0.8609 | 0.1360 → 0.1691 | — |
| **0.003 (selected on val)** | **0.8822** | 0.8591 | 0.1360 → 0.1699 | +0.0008 |
| 0.01 | 0.8794 | 0.8540 | 0.1359 → 0.1705 | +0.0013 |
| 0.03 | 0.8713 | 0.8573 | 0.1354 → 0.1689 | −0.0002 |
| 0.1 | 0.8701 | 0.8485 | 0.1343 → 0.1713 | +0.0021 |
| 0.3 | 0.8571 | 0.8352 | 0.1316 → 0.1550 | −0.0141 |
| 1.0 | 0.1018 | 0.1026 | (collapsed) | — |

Plain ERM for the same encoder, protocol and seed: val 0.8759, test 0.8587.

**Tuned properly, the method does exactly nothing.** Weight selected on
validation gives test 0.8591 against ERM's 0.8587 — **+0.0004**, an order of
magnitude below the ~0.002 same-code-path gap noted in §9 and two orders below
the seed spread. And the reason it does nothing is visible in the penalty
column, which is why logging the objectives' aux dict was worth the five-line
change: **at the selected weight the OT penalty is not reduced at all** (0.1699
against 0.1691 unpenalized). This is not a method trading accuracy for
invariance and landing at a wash. It is a method that is switched off.

The penalty only moves at λ = 0.3, where it falls 8.3% and costs 0.0257 of
macro-F1; one step further and the embedding collapses. So the useful window —
penalty down, accuracy held — does not exist anywhere in three orders of
magnitude.

**What I would have concluded without the sweep, and why it would have been
wrong.** The original table showed `sinkhorn` at 0.1026, a delta of −0.7562, in
a row that a reader would take as "optimal transport fails badly on this
benchmark". That reading is unearned: it is one untuned default, and agy was
right to call it a strawman. The corrected claim is narrower, better supported
and less dramatic: entropic OT between lot embeddings, swept over three orders
of magnitude and selected honestly, is indistinguishable from ERM, and its
penalty term is inert at every weight that trains.

**Note the interaction with §4.** These sinkhorn cells used the `hash32` domain
definition, whose 32 buckets have a mean pairwise label TV of 0.0208. An OT
penalty between near-identical clouds having nothing to reduce is *exactly what
the penalty column shows*. So this result and the domain-definition result are
the same finding seen from two directions, and the honest statement is that
entropic OT is inert **on domains defined this way** — which is one more reason
the `domain_def` sweep, now the next stage to run, is the most important thing
queued. I am not going to claim OT is dead on this corpus until it has been
given domains that differ.

The row stays in the tables with the collapse visible and the sweep beside it.
A table that quietly drops its embarrassments is worse than one that explains
them.

### 11. Decision 3 is closed without a GPU run: the label-aware guard never fires

Both external critics flagged `_stratified_group_split` reading candidate
groups' labels. I logged it in §4 and §7 as real, mis-named as leakage, and
needing a measurement, and put it to the owner in `WEEKEND.md` as a decision
with a "cheap, one flag and a 20-cell re-run" recommendation. The re-run turns
out not to be needed, because the cheaper question settles it.

`scripts/label_blind_check.py` adds `label_blind=True` to `split()` — dropping
the guard entirely — and compares the two splits directly, no model involved:

| protocol | seeds where the guard fired | seeds where the split differed | seeds where blind lost a class |
|---|---|---|---|
| `lot` | **0 / 10** | **0 / 10** | 0 / 10 |
| `size` | **0 / 10** | **0 / 10** | 0 / 10 |

The guard never rejects a single group, and the label-aware and label-blind
partitions are identical wafer for wafer at every seed tested. The magnitude of
the bias is not small; it is **zero** on this corpus at a 25% holdout.

**And the reason is structural, not luck**, which matters because "it did not
happen in ten seeds" is a weak claim on its own. The guard rejects a group only
when that single group holds *all* the remaining training examples of some
class. Measured concentration:

| class | n | max share in one lot | max share in one geometry |
|---|---|---|---|
| none | 147,429 | 0.0002 | 0.1077 |
| Center | 4,294 | 0.0054 | 0.5242 |
| Donut | 555 | 0.0414 | 0.4216 |
| Edge-Loc | 5,189 | 0.0046 | 0.0912 |
| Edge-Ring | 9,680 | 0.0026 | 0.2193 |
| Loc | 3,593 | 0.0070 | 0.0827 |
| Near-full | 149 | 0.0268 | 0.1879 |
| Random | 866 | 0.0289 | 0.1443 |
| Scratch | 1,193 | 0.0101 | 0.0671 |

No lot holds more than 4.1% of any class; even the rarest class, `Near-full` at
149 wafers, is spread over 137 lots and 30 geometries. The condition the guard
tests for is unreachable. It is dead code with respect to this corpus.

**The honest statement**, which is narrower than either "the critics were wrong"
or "we fixed it": the code path *is* label-aware and a reviewer is right to
object to it on principle, because on a corpus where some class sat mostly in
one lot it would bias the split and the bias would flatter macro-F1. On WM-811K
it does nothing. The flag stays so the claim is checkable rather than asserted,
`WEEKEND.md` drops this from the decisions list, and the 20 cells it would have
cost go to the queue instead.

Worth noting what this cost to find out: about four minutes of CPU. I had
written it into the Monday hand-off as a decision needing a human, on the
strength of two reviewers agreeing it mattered. Two reviewers agreeing is not
evidence, and the cheap check should have come before the hand-off entry.

### 12. The next question the forward-only confound raises, and the control for it

§8 established that `lot_time` tests on 19 geometries against 338 in training,
with 14.15% of its test wafers of a geometry never trained on. The queued
seen/unseen decomposition splits its drop by that boundary. But there is a
second, competing explanation it does not address:

> **H12:** the forward-only drop is not about time or about *unseen* geometry
> at all — a *narrow* geometry slice is simply a harder test set, and any
> protocol restricted to those 19 geometries would score near 0.70 even with
> every one of them seen in training and no temporal ordering whatsoever.

The control is nearly free and needs no new training: take the `lot`-protocol
model, whose training set covers those geometries and whose split has no
temporal structure, and score it on the subset of *its own* test set falling in
those 19 geometries. Measured feasibility, at three seeds: 11,908 / 11,789 /
11,474 wafers, 26.5–27.5% of each `lot` test set, with all 9 classes present, so
the comparison is not confounded by a missing class.

Reading, fixed in advance so the result cannot be reinterpreted afterwards:

* `lot` restricted to those geometries lands near the full `lot` figure →
  narrowness is not the cause, and the forward-only drop belongs to time plus
  unseen geometry;
* it lands near the `lot_time` figure → the geometry slice explains the drop on
  its own, and "forward-only time costs 20 points" is close to wholly wrong;
* in between → the drop is a compound and this measures the geometry share.

`run_bench.py` now scores `test_in_lot_time_geometries` on every cell, so the
answer arrives with the backfill already queued rather than needing a stage of
its own.

### 13. "Active learning lost to random" compared strategies that had bought a fifth of the data

This is one of the three results the paper was going to be built on, and I had
never looked at it. The numbers, from `runs/active_learning.json`, at a 400-lot
budget: random 0.7241, diverse 0.6581, coreset 0.5938, entropy 0.5698. Read as
"per-wafer uncertainty ranking does not survive whole-lot acquisition", which is
a clean and quotable finding.

The same file records `wafers_mean`, which nobody had read:

| lot budget | 20 | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|
| random, wafers labelled | 323 | 728 | 1,491 | 3,012 | **6,284** | 12,526 |
| entropy | 323 | 355 | 411 | 544 | **1,104** | 2,464 |
| coreset | 323 | 370 | 501 | 713 | **1,820** | 2,900 |
| diverse | 323 | 405 | 524 | 794 | **1,344** | 2,478 |

At the 400-lot budget the heuristics trained on **17.6%, 29.0% and 21.4%** of
random's wafers. Mean lot size bought: random 15.7 wafers, entropy 2.8. The
comparison was never between acquisition strategies; it was between a strategy
with 6,284 labels and strategies with 1,100–1,800.

**The mechanism, and it is not subtle once seen.** `lot_scores`
(`scripts/active_learning.py:79`) scores a lot as the **mean** of its wafers'
scores, and the acquisition takes the top-scoring lots. The maximum of noisy
means favours small samples: a 2-wafer lot's mean is one or two draws and can
land anywhere in the tail, while a 25-wafer lot's mean regresses to the pool
average. So "take the highest-entropy lots" is, in substantial part, "take the
smallest lots". Random sampling has no such bias, which is exactly why it looked
good. This is a selection artefact of the scoring rule, not a property of
uncertainty sampling.

**What happens on the axis that was implicitly varying.** Re-plotting the stored
curve against wafers labelled (`scripts/al_budget_check.py`, linear
interpolation between measured points, no extrapolation):

| wafers labelled | 322 | 750 | 1,179 | 1,607 | 2,035 | 2,463 |
|---|---|---|---|---|---|---|
| random | 0.3794 | 0.5116 | 0.5908 | 0.6500 | 0.6556 | 0.6612 |
| entropy | 0.3794 | 0.5693 | 0.5786 | 0.6285 | 0.6785 | **0.7285** |
| coreset | 0.3794 | 0.5837 | 0.5877 | 0.5918 | 0.6110 | 0.6451 |
| diverse | 0.3794 | 0.5762 | 0.6376 | 0.6609 | 0.6655 | 0.6702 |

Random wins only at 322 wafers, where every strategy holds the identical random
seed set by construction and the comparison is vacuous. At every other matched
volume a heuristic is ahead. Directly from the stored table, without any
interpolation: **entropy reaches 0.7285 ± 0.0134 on 2,464 wafers; random needs
6,284 wafers to reach 0.7241 ± 0.0049** — the same accuracy for 2.6x the labels.

**What I am claiming and what I am not.** The data-volume confound is certain:
it is arithmetic on the stored file, no noise involved, and it means the
published ranking does not support the sentence it was written to support. The
*reversal* is not certain. It rests on three seeds, on interpolation between six
points, and on curves whose per-point standard deviations run to 0.035. It is
enough to require the experiment and not enough to be the result.

So `active_learning.py` gains `--budget-unit {lots,wafers}` and
`scripts/al_wafer_budget.sh` re-runs the whole grid with every strategy stopped
at the same supervision volume, queued as the last stage of `chain_rest.sh`. It
also now records the lots it bought and their sizes, so the next person can
diagnose a selection bias without re-running anything — which is what cost this
finding a day.

**The part that is a genuine decision rather than a bug, and I want it stated
plainly because it is the most interesting thing here.** The two cost models
disagree, and neither is wrong:

* if a metrology slot costs one **lot** regardless of how many wafers it holds,
  the original axis is right, and the finding is that these heuristics waste the
  slot by selecting near-empty lots — actionable, and a genuine failure;
* if the cost is per **wafer measured**, the wafer axis is right and the
  heuristics are ahead.

Measured in the smoke run of the new code: to acquire ~780 wafers, random needed
**50** lots and entropy needed **360**. Same data volume, 7x the slots. A real
fab pays something of both, so the honest presentation is both curves side by
side with the cost model named, and the headline "active learning lost to
random" is withdrawn pending the wafer-budget run.

I should have read `wafers_mean` the first time I looked at that table. It was
in the file, in the same object as the number I was quoting.

### 14. Partial: giving the objectives a real domain makes one of them move, downward

`domain_def_sweep.sh` at 14/35 cells. `cnn_bn` on `lot`, deltas against ERM
measured under the *same* domain definition so the comparison is internal:

| objective | domain = `lot % 32` (label TV 0.021) | vs ERM | domain = production decile (TV 0.182) | vs ERM |
|---|---|---|---|---|
| `erm` | 0.8522 ±0.0069 (n=3) | — | 0.8514 ±0.0073 (n=3) | — |
| `group_dro` | 0.8535 ±0.0063 (n=3) | +0.0013 | 0.8257 ±0.0139 (n=3) | **−0.0257** |
| `irm` | 0.8418 ±0.0094 (n=3) | −0.0103 | 0.8476 ±0.0052 (n=2) | −0.0038 |

`coral`, `dann`, `mixup_domain`, `hsic` pending.

The null control holds: `erm` never reads the domain label and its two columns
differ by 0.0008, inside the ~0.002 same-code-path floor, so the machinery did
not change ERM and the other rows are differences in the objective.

GroupDRO is the first objective to move once the domain means something, and it
moves *down* by 0.0257 against a half-range of 0.0139 — roughly two half-ranges,
so a real effect rather than noise. This is worth being precise about, because
it is the opposite of a rescue: H4 asked whether the ERM-equivalence was an
artefact of a degenerate domain definition, and the early answer is that the
artefact was real *and* fixing it does not produce a method that beats ERM. It
produces a method that loses. That is a considerably stronger negative result
than the one it replaces — "these objectives tie with ERM because they were
switched off" becomes "switched on, at least one of them is actively harmful" —
and it is consistent with the `size` column, where the hash was never degenerate
and GroupDRO was already −0.18 to −0.27.

I will not write that conclusion until the remaining four objectives land.

### 15. The representation table has no error bars, and almost none of its ordering is resolvable

The paper's framing table — six representations across four protocols — was
measured at one seed per cell. Where seeds have since been run the half-ranges
are 0.0044–0.0096 on `lot`, 0.0045–0.0070 on `lot_time` and 0.031–0.036 on
`size`. The gaps in the table are the same size.

`report.py` now prints the verdict for every adjacent pair, non-parametrically:
three seeds do not support a p-value, but whether one cell's worst seed beat the
other's best is a fact about what was observed. Of **18 adjacent pairs across
the four protocols, not one is resolved**: three have overlapping seed ranges and
fifteen have no error bar at all. On `lot`, the entire ordering
`rpca_cnn > cnn_gn > spectral > cnn_bn > feat` sits inside the noise; only the
die-graph GNN, 0.086 below the next cell, is clearly last.

Two claims from the project summary are directly affected:

* **"spectral 0.8538, CNN+BatchNorm 0.8587"** — the two are 0.0017 apart on the
  three-seed mean, against a `cnn_bn` half-range of 0.0069. Not a ranking.
* **"GroupNorm beats BatchNorm everywhere"** — checked as its own claim, since
  `spectral` sits between them and it never appears as an adjacent pair. On
  `lot` every GroupNorm seed does beat every BatchNorm seed, which is the right
  test, but the two ranges are separated by **0.0004** — below the run-to-run
  floor of two identical invocations. So the direction is consistent and the
  margin is not meaningful. On `size` and `lot_time` the gap is much larger
  (+0.0784, +0.0545) but BatchNorm is single-seed there, so those are unbarred
  too.

The mechanism argument for GroupNorm remains good independent of the numbers —
BatchNorm mixes statistics across whatever is in the batch and a batch here
spans lots, which is a domain leak by construction. But a mechanism is a reason
to expect an effect, not evidence of one, and the table was being read as
evidence.

**Change made.** `scripts/backfill_metrics.sh` now runs all 22
(protocol, representation) pairs at three seeds rather than only re-running the
seeds that happen to exist — 66 cells, about 35 minutes on two GPUs, estimated
from the stored per-cell wall times. It was already queued as the next stage
after the domain-definition sweep, so this costs no extra scheduling.

**What I expect, written down now.** Most of these gaps will stay unresolved,
because 0.01 gaps against 0.007 half-ranges do not separate at three seeds. The
useful outcome is not a cleaner ranking; it is being able to say which
comparisons this corpus can support at all, and the honest answer is likely to
be "the GNN is worse, and nothing else is distinguishable". A benchmark that
cannot rank its own baselines is a finding about the benchmark, and a more
useful one than a leaderboard.

### 16. The reproducibility floor was measured from one pair of runs, and was 4.5x too small

The backfill gate refused. It compared a fresh re-run of `lot/cnn_gn/erm/s0`
against the stored value, found them 0.0082 apart, and declared that larger than
the "run-to-run floor" of 0.0012 it had just measured. That was the right call
for the wrong reason: **0.0012 was the difference between exactly two runs**, a
one-sample estimate of a spread, and I had multiplied it by three and called it
a tolerance.

Six identical invocations of the same cell (`scripts/determinism_repeats.sh`):

```
0.8723  0.8760  0.8767  0.8767  0.8772  0.8778
range 0.0054   stdev 0.0019
```

So the floor is **0.0054**, four and a half times the pair-based estimate. A
gate whose tolerance is estimated that badly refuses correct backfills, and
would as happily admit wrong ones.

But the stored value, 0.8671, is **outside** all six repeats — 0.0052 below the
nearest. Six of six above it is not symmetric noise, so the gate's refusal
stood and something had to be found.

**What it was not: my code.** I extracted `scripts/run_bench.py` and the whole
`wts` package at commit `373b9cc` — before every change made this weekend — into
a scratch directory and ran the identical cell three times on my own GPUs:

```
old code, GPUs 0/1:  0.8742  0.8784  0.8753
new code, GPUs 0/1:  0.8723 ... 0.8778
stored, Friday:      0.8671
```

The old code lands inside the new code's range. Nothing I changed moved the
number. The difference is environmental — the original sweep ran on GPUs 2 and 3
(`sweep.sh` defaults to them) and this weekend's work is confined to 0 and 1 by
the GPU lease, so the comparison could not be run on the original hardware
without taking devices that belong to another session. I did not take them.

**The finding this actually produces, which is larger than the bug it was
chasing.** The *total* reproducibility spread of this benchmark, across the
hardware and sessions it has actually been run on, is around 0.01 — the 0.0054
within one session plus a further ~0.005 offset between sessions. The seed
spreads I have been quoting all weekend (±0.004–0.010 on `lot`) are an
**underestimate of total variability**, because all three seeds of any cell were
run back-to-back on the same two GPUs. Seed spread measures the seed; it does
not measure the pipeline.

### 17. Applying the floor: nothing separates from ERM, including the one thing that looked like it did

`report.py` now runs every verdict through one helper that requires two things:
the seed ranges must not overlap, **and** the margin between them must exceed
the measured run-to-run floor. A margin below the floor is not a separation
however cleanly the seeds sort.

Re-reading the completed domain-definition sweep (`cnn_bn`, `lot`, 3 seeds
each, deltas against ERM under the *same* domain definition):

| objective | `lot % 32` (TV 0.021) | vs ERM | production decile (TV 0.182) | vs ERM | verdict under real domains |
|---|---|---|---|---|---|
| `erm` | 0.8522 ±0.0069 | — | 0.8514 ±0.0073 | — | — |
| `group_dro` | 0.8535 ±0.0063 | +0.0013 | 0.8257 ±0.0139 | −0.0257 | margin 0.0035 < floor 0.0054 |
| `hsic` | 0.8527 ±0.0080 | +0.0005 | 0.8488 ±0.0076 | −0.0026 | ranges overlap |
| `irm` | 0.8418 ±0.0094 | −0.0103 | 0.8465 ±0.0052 | −0.0049 | ranges overlap |
| `coral` | 0.8468 ±0.0090 | −0.0053 | 0.8443 ±0.0098 | −0.0071 | ranges overlap |
| `dann` | 0.8517 ±0.0080 | −0.0005 | 0.8454 ±0.0132 | −0.0060 | ranges overlap |
| `mixup_domain` | 0.8398 ±0.0108 | −0.0124 | 0.8371 ±0.0138 | −0.0143 | ranges overlap |

Last turn I wrote that GroupDRO "is the first objective to move once the domain
means something, and it moves down… roughly two half-ranges, so a real effect
rather than noise." That was wrong, and it was wrong because I was comparing
against a seed spread rather than against the pipeline's own reproducibility.
The margin between GroupDRO's seed range and ERM's is 0.0035, below the 0.0054
floor. I withdraw it.

**H4 is answered, and the answer is cleaner than either outcome I anticipated.**
The ERM-equivalence was *both* an artefact of a degenerate domain definition
*and* true anyway: given a domain vocabulary carrying nine times the label
shift, not one of seven borrowed objectives can be distinguished from ERM at the
floor. The original claim was unearned — the experiment as run could not have
shown otherwise — and the corrected experiment reaches the same conclusion
honestly. That is the version worth publishing.

### 18. The gamma=0 control earned its place

`focal --focal-gamma 0` is exactly cross-entropy, asserted bit-for-bit in
`tests/test_smoke.py`, and it was put in the sweep so that a movement could be
attributed. It caught something:

```
ERM cells        0.8591  0.8671  0.8680     mean 0.8647
focal g=0 cells  0.8680  0.8781             mean 0.8730
```

The two are the same computation, and the focal batch sits **+0.0083** above the
ERM cells. Under the naive range-overlap test this read as a clean separation —
"focal γ=0 beats ERM by 0.008" — which is impossible. It is a batch offset of
exactly the size described in §16.

So the honest comparison is each γ against **γ=0, run in the same batch**:

| cell | seeds | vs γ=0 | verdict |
|---|---|---|---|
| γ=0.0 | 0.8680, 0.8781 | — | — |
| γ=0.5 | 0.8758, 0.8760 | +0.0028 | ranges overlap |
| γ=1.0 | 0.8710, 0.8748 | −0.0001 | ranges overlap |
| γ=2.0 | 0.8654, 0.8749 | −0.0029 | ranges overlap |
| γ=5.0 | 0.8694, 0.8795 | +0.0014 | ranges overlap |

**Focal loss does nothing on this corpus**, which is what
`scripts/longtail_sweep.sh` recorded as its prior before running. Against the
*wrong* baseline it would have read as +0.008 to +0.011 and looked like the
weekend's one positive result. Class-balanced weighting likewise overlaps ERM.

The general lesson, which I want stated because it cost nothing and saved a
false claim: when a sweep is run as a batch, its own no-op setting is a better
baseline than a cell measured in a different batch, and every sweep should carry
one.

### 19. What survives the floor

Not everything collapses. Checked against the same 0.0054:

* **SSL pretraining is worse than random initialization**, at every learning
  rate, with margins of 0.035 to 0.136 — six to twenty-five times the floor.
  And my recorded prediction was wrong in an informative direction: I expected
  that if the deficit were a recipe mismatch the gap would *close* at low LR.
  It widens — −0.0504 at 2e-3, −0.0935 at 1e-3, −0.1102 at 5e-4, −0.1464 at
  2e-4. The confound is refuted, not merely unsupported.
* **The die-graph GNN is worse than everything else**, by 0.086 on `lot`.
* **The RPCA fourth channel is matched by zeros** — that was already an
  overlap result and the floor only strengthens it.

### 20. Active learning on a wafer budget: the conclusion reverses, and the real question is the price of a metrology slot

The re-run finished after the k-center memory fix. Same pool, same test split,
same seeds; the only change is that every strategy stops at the same number of
labelled **wafers** instead of the same number of lots. Wafer counts come out
equal by construction (387–400, 792–800, …), so data volume is held fixed:

| macro-F1 at | 400 | 800 | 1,600 | 3,200 | 6,400 wafers |
|---|---|---|---|---|---|
| random | 0.4274 | 0.4923 | 0.5942 | 0.6697 | 0.7257 |
| **entropy** | **0.5456** | **0.5954** | **0.7003** | **0.7432** | **0.7770** |
| coreset | 0.5539 | 0.5904 | 0.6642 | 0.7151 | 0.7681 |
| diverse | 0.5010 | 0.5610 | 0.6332 | 0.7364 | 0.7434 |

Seed standard deviations run 0.004–0.047; entropy's advantage runs 0.05–0.13,
two to ten times that and far above the 0.0054 reproducibility floor. **At
matched supervision volume, uncertainty sampling beats random at every budget
measured.** The published claim was the opposite, and it was an artefact of
counting the budget in lots while the heuristics systematically bought the
smallest lots available.

**But the reversal is not the finding either, and I want to be careful not to
replace one over-claim with its mirror image.** The lots each strategy needed to
buy those wafers:

| lots needed for | 400 | 800 | 1,600 | 3,200 | 6,400 wafers |
|---|---|---|---|---|---|
| random | 25 | 54 | 103 | 205 | **397** |
| entropy | 95 | 365 | 666 | 1,008 | **1,355** |

Entropy needs **3.4x the metrology slots** for the same number of measured
wafers. So both of these are true, measured on the same corpus:

* priced per **lot**, random acquisition wins and the heuristics waste slots on
  near-empty lots;
* priced per **wafer**, entropy wins by 0.05 to 0.13.

Neither is a fact about active learning. Both are facts about active learning
*plus a cost model*, and the cost model was never stated — it was smuggled in as
"budget = 400" in a table header. That is the actual methodological finding, and
it generalizes past this benchmark: an acquisition experiment whose budget unit
is not the thing being paid for measures the budget unit, not the acquisition.

Both curves are now in `RESULTS.md` with the cost model named at each. The
headline is withdrawn and not replaced; `WEEKEND.md` Decision 4 asks the owner
which price is real, and it is now quantified on both axes rather than posed as
an open question.

**A note on how this was nearly missed twice.** The first wafer-budget run died
two strategies in: the k-center distance materialized a |pool| x |chosen| x dim
array, was killed by the OOM killer with no traceback, and the stage logged
`=== al wafer budget done ===` and wrote no JSON. Nothing failed loudly. That is
the third silent-failure mode this weekend — after the redirect into a
nonexistent directory that dropped three cells, and the reporter key collision
that would have overwritten seeds — and all three shared the property that the
harness reported success. The fix is chunked accumulation, verified against the
naive formula to 2.9e-06.

### 21. The handed-down target, answered: 0.95 is not reachable here, and focal is not the reason

The target was *multi-label F1 >= 0.95 under scarce labels, with focal loss*.
It has now been measured rather than argued about, on MIXED-SYNTH — the overlay
construction built because MixedWM38 could not be obtained from this machine.

| protocol | best loss | macro-F1 @tuned | micro-F1 @tuned | subset acc. | vs 0.95 |
|---|---|---|---|---|---|
| optimistic (`iid` sources) | focal | 0.8427 | 0.8524 | 0.8866 | **−0.1073** |
| honest (`lot`-disjoint sources) | focal | 0.8177 | 0.8440 | 0.8805 | **−0.1323** |

Every reading misses, and the closest one is the optimistic split of a dataset
that is by construction *easier than the real task*: overlaid patterns do not
interact, so a scratch crossing an edge ring changes neither signature, and the
real MixedWM38 problem is harder than this by an unmeasured amount.

**Focal contributes nothing, and it was named in the target as if it were the
mechanism.** Its seed range overlaps plain BCE's on both protocols (iid 0.8427
vs 0.8427; lot 0.8177 vs 0.8136). Positive-class weighting is clearly worse
(−0.0147, −0.0071). This is the second independent test of the same claim: the
single-label sweep over gamma in {0, 0.5, 1, 2, 5}, measured against its own
bit-exact `gamma = 0` control, also moved nothing.

**Why published figures near 0.95 exist and this one does not.** Three
constructions inflate a "multi-label F1" on wafer data, and all three are
avoided here:

1. counting the defect-free majority as a label — it is ~85% of the corpus,
   trivially separable, and it drags any average upward. Here a defect-free
   wafer is the all-zero target and contributes true negatives only;
2. reporting weighted- or micro-F1 over a set that includes that majority;
3. an iid split. Ours is reported, labelled optimistic, and is 0.8427.

The sibling repository's 0.980 weighted-F1 is an instance of (1) and (2) and its
README says so at the point the number appears; its macro-F1 on the same run is
0.897. None of these are the same quantity and none should be compared.

**The shortfall is where it has been all weekend.** Per-class F1 in the best
lot-disjoint cell: Scratch 0.6888, Donut 0.7284, Loc 0.7564, then Random 0.8299
up to Edge-Ring 0.9296. `Scratch` and `Loc` are last here exactly as they are
last on the single-label task. The multi-label framing changes nothing about
which patterns are hard, which is itself worth reporting: a one-die-wide line is
hard because it is a one-die-wide line, not because of how the label is encoded.

**What I would tell the owner.** The gap to 0.95 is not a tuning gap and it is
not a loss-function gap — five values of gamma, class-balanced weighting and
positive weighting have all now been measured and none of them moves it. If
0.95 is required, the remaining honest levers are input resolution (a scratch is
a few dies wide and 64x64 blurs it), a larger backbone, and real MixedWM38 data.
The first of those is the one this corpus most obviously invites and it has not
been tried.

### 22. I proposed a lever in the last commit and refuted it in this one

The §21 write-up ended: *"the remaining honest levers are input resolution (a
scratch is a few dies wide and 64x64 blurs it), a larger backbone, and real
MixedWM38 data."* The parenthesis is a claim, it went into a commit message and
into the paper's forward-looking text, and it is **false on this corpus**.
Checking it cost one line:

| class | n | median max dimension | fraction downsampled by 64x64 |
|---|---|---|---|
| Scratch | 1,193 | 41 | 0.155 |
| Edge-Ring | 9,680 | 53 | 0.146 |
| Loc | 3,593 | 36 | 0.069 |
| Donut | 555 | 42 | 0.061 |
| none | 147,429 | 34 | 0.008 |

**97.73% of wafers are *upsampled* to reach 64x64**; 2.01% are downsampled. The
median wafer's larger dimension is 34 dies. The resize adds pixels; it does not
remove detail. Raising the input resolution to 128 would interpolate, not
recover — there is nothing to recover for 49 wafers in 50.

`Scratch` is the most-downsampled class at 15.5%, which is consistent with the
intuition having *some* basis, and still leaves 84.5% of scratches upsampled. It
is not the explanation for a per-class F1 of 0.69.

**Where the claim came from, which is the part worth recording.** It is a
near-quotation of the sibling repository's hand-off note, which lists "higher
input resolution (a scratch is a few dies wide and 64x64 blurs it)" as its first
candidate for closing the same gap. That repository resamples to 64x64 from the
same corpus, so the statement is very likely wrong there too — but I have not
measured it there and am not asserting it. What I did was carry a plausible
mechanism across from a neighbouring document and restate it as if it were
established here. That is exactly the failure mode the "no number unless a run
produced it" rule exists to prevent, in its qualitative form: **no *mechanism*
either, unless a measurement supports it.** A sentence does not become true by
having been written down next door.

Corrected in `paper_draft.md` §7.2, which now presents the check and the
refutation rather than the intuition, and the measurement is in
`runs/corpus_stats.json` so the table regenerates.

**What this leaves.** The long tail is not explained by class imbalance (focal
over five gammas against a bit-exact control: nothing; class-balanced weights:
nothing; positive weighting: worse), and not by input resolution. The remaining
candidates I would rank: global average pooling discarding the localization a
thin structure needs; genuine label ambiguity in weak instances, which the
sibling repository measured directly as Loc/Scratch leaking into `none` rather
than into each other; and simply too few examples of a thin, high-variance
pattern. Only the first is cheap to test and it is a real architectural
hypothesis rather than a tuning one.

### 23. A quarter of the forward-only drop is the test set being narrow, not time

H12 has its first clean answer, and it did not need a new protocol — only a
different subset of an existing test set. `run_bench.py` now scores every cell
on the 19 geometries that `lot_time` tests on, and the `iid` cells from the
single-session grid are the cleanest possible control: a random split, every one
of those geometries seen in training, only 28 of 43,209 test wafers of an unseen
geometry, and no temporal structure whatever.

| representation | full test | restricted to `lot_time`'s 19 geometries | cost |
|---|---|---|---|
| CNN (GroupNorm) | 0.8820 | 0.8356 | **−0.0464** |
| CNN (BatchNorm) | 0.8623 | 0.8040 | **−0.0583** |
| spectral operator | 0.8578 | 0.7872 | **−0.0706** |
| descriptors + MLP | 0.8532 | 0.7673 | **−0.0859** |

All nine classes are present in the restricted subset, so the matched and
unmatched averages are identical and the comparison is exactly like-for-like —
the trap I built the class-matching for did not fire here, which is worth
knowing.

**That slice is intrinsically hard, by 0.046 to 0.086, with no time and no
unseen geometry involved.** Every one of those is many times the 0.0054
reproducibility floor. For the GroupNorm CNN the full drop from `iid` to
`lot_time` is about 0.18, so **roughly a quarter of it is accounted for before
the forward-only ordering does anything at all**. The same restriction on the
`lot` protocol costs 0.048 to 0.101, so the effect is a property of the
geometries, not of a protocol.

The corrected claim, which I will not sharpen further until the single-session
`lot_time` cells land: forward-only deployment costs about 0.18, of which a
quarter is that the wafers you meet later are of geometries that are harder to
classify, an unmeasured further share is that some of them were never trained
on, and the remainder is drift. "Forward-only time costs 20 points" was never
what the experiment measured.

### 24. What is left of the long tail, and the one hypothesis still standing

Two explanations have now been measured and discarded:

* **class imbalance** — focal over gamma in {0, 0.5, 1, 2, 5} against its own
  bit-exact `gamma = 0` control: every range overlaps. Class-balanced weights:
  overlaps. Positive weighting on the multi-label task: clearly worse. Three
  corrections, none of which moves `Scratch`.
* **input resolution** — refuted in §22; 97.7% of wafers are upsampled to reach
  64x64, so there is nothing for a finer grid to recover.

The hypothesis still standing is architectural, and it is the one the encoder
makes most obviously. `CnnResized.embed` is a global average over the final
feature map. A `Scratch` is a thin *connected line* of failed dies; averaged
over the wafer it is nearly indistinguishable from a slightly elevated
background failure rate, because the mean is exactly the statistic that discards
the fact that the failures form a line. `Near-full` and `Edge-Ring`, which are
the classes the model is best at, are precisely the ones a mean describes well.

> **H24:** replacing global average pooling with something that preserves the
> extreme a thin structure produces will move `Scratch` and `Loc` specifically,
> and will do little to macro-F1, because seven of the nine classes are already
> well served by a mean.

`scripts/pooling_sweep.sh` runs `cnn_gn` on `lot` at three seeds each for:

* `mean` — the original;
* `meanmax` — concatenate mean and max, the treatment;
* `meanmean` — concatenate the mean with itself: **identical parameter count,
  no extra information**.

The control is the whole design. The RPCA fourth channel read as +0.0142 until
it was run against a channel of zeros and turned out to be worth exactly what
nothing was worth; `meanmean` asks that question before the claim rather than
after it. `tests/test_smoke.py` asserts the control matches the treatment in
parameter count and that its two halves are identical, so it cannot quietly stop
being a control.

Read `Scratch` and `Loc` F1, not macro-F1 — a nine-class macro average is mostly
about the other seven, and if the hypothesis is right macro-F1 is the wrong
place to look for it. Queued behind the single-session grid.

### 25. The paper's headline table was still reporting deltas without verdicts

`paper_draft.md` section 3 listed every objective's delta against ERM and left
the reader to decide what counted. With a measured floor of 0.0054 in hand that
is no longer a defensible way to present it, so the generator now attaches a
verdict to every row — ranges disjoint *and* margin above the floor — and counts
them:

> *Of the 6 cells on this protocol that have an error bar at all, **0** clear
> the floor.*

That sentence is computed, not written, so it cannot go stale the way the three
things I fixed alongside it had:

* the section heading still said the ERM-equivalence was "partly an artefact of
  our own domain definition", written before the sweep completed. It now says
  the sharper thing: the first version of the experiment could not have shown
  otherwise, and the second reaches the same conclusion honestly;
* the `sinkhorn` note still said the weight sweep *would* run and the row
  *would* be removed if nothing worked. The sweep ran two turns ago. The note
  now carries its answer — validation picks `--ot-lambda 0.003`, at which the OT
  penalty ends at 0.1699 against 0.1691 unpenalized, so the objective is not
  trading accuracy for invariance, it is switched off;
* the `anchor` and `logit_adjust` rows showed deltas of +0.0115 and −0.0352 with
  nothing to indicate they are single-seed.

### 26. The last claim in this repository with no error bar is also its largest

Adding verdicts made a gap obvious that I had been quoting past all weekend.
**Every objective cell on the `size` protocol is single-seed** — the whole table
reads "no error bar" — and two of those cells are the biggest effects anywhere
in this work: GroupDRO at −0.18 to −0.27 and logit adjustment at −0.08 to −0.10.

I have leaned on them repeatedly. In §4 I wrote that the `size` half of the
negative result "stands as measured" because the hash was not degenerate there;
in §5 that GroupDRO's −0.18 to −0.27 is "five to eight times the seed half-range,
so those remain real effects". That reasoning used the seed half-range measured
on **`cnn_gn` under ERM**, not on the cells in question, which have no seeds at
all. It is the same substitution that made `focal γ=0` look like a +0.0083 win:
comparing against a spread measured somewhere else.

And `size` is the worst protocol on which to do it. Its measured half-range is
0.031–0.036, an order of magnitude above `lot`'s, because each seed holds out
*different geometries* and geometries are heterogeneous. At n = 1 a −0.27 and a
−0.03 are equally unfalsifiable; the first merely looks safer.

> **H26:** the large negative effects on `size` survive three seeds. GroupDRO's
> −0.27 is roughly eight times that protocol's half-range and I expect it to
> hold; logit adjustment's −0.08 is between two and three times it and I am much
> less confident. If either fails, the domain-generalization result loses its
> last surviving *directional* claim and becomes uniformly "nothing separates
> from ERM anywhere", which would be cleaner and duller.

`scripts/size_objectives_seeds.sh` runs `{cnn_bn, feat} x {erm, group_dro,
logit_adjust, irm, coral, dann, mixup_domain} x 3 seeds` — 42 cells, all in one
session with ERM re-run alongside so every delta is internal to that session.
Queued third, behind the representation grid and the pooling sweep.

Writing the prediction down first is the point. If GroupDRO holds I want to have
said so beforehand, and if it does not I want it on record that the weekend's
most-quoted surviving number went the way of the others.

### 27. The single-session grid does not reduce spread, and I should not have assumed it would

Two turns ago I launched a 66-cell grid on the reasoning that mixing Friday's
seed 0 with this weekend's seeds 1–2 folds a session offset into what is
presented as seed variance. The `lot` cells have now landed for both encoders
that have a mixed-session counterpart:

| cell | mixed-session range (n=3) | one-session range (n=3) | ratio |
|---|---|---|---|
| `lot` / cnn_bn | 0.0139 | 0.0089 | 0.64 |
| `lot` / cnn_gn | 0.0088 | 0.0145 | 1.64 |

One went down, one went up, by almost reciprocal factors. **The design does not
demonstrably reduce spread**, and with two cells at n=3 each the comparison
cannot resolve it — a range from three samples is a very noisy statistic, which
is the whole reason this weekend has been what it has been.

What the offset argument still establishes is the *offset*, which is real and
measured: the stored Friday cell sits outside all six repeats of itself. What it
does not establish is that within-session seeds are tighter. The grid is still
worth having, because an internally consistent table is worth having, but I have
corrected the claim in the script's own header rather than leaving a rationale
that the data does not support.

### 28. My verdict criterion commits the error it was built to prevent

In §26 I criticised myself for judging `size` deltas against a seed half-range
measured on `cnn_gn` under ERM — "comparing against a spread measured somewhere
else". The floor-based criterion introduced in §17 does exactly that, and I did
not notice for four entries.

The floor is **one number, 0.0054, measured on one cell** (`lot/cnn_gn/erm`),
and it has been applied to every verdict on every protocol. Observed seed ranges
by protocol, over cells with three seeds:

| protocol | cells | seed range | median |
|---|---|---|---|
| `lot` | 6 | 0.0088 – 0.0192 | 0.0139 |
| `lot_time` | 2 | 0.0090 – 0.0139 | 0.0139 |
| `iid` | 4 | 0.0033 – 0.0285 | 0.0152 |
| **`size`** | 2 | **0.0692 – 0.0718** | 0.0718 |

`size` is an order of magnitude looser than `lot`, for a reason that is a
property of the protocol rather than an accident: each seed holds out *different
geometries*, and geometries are heterogeneous. Judging a `size` delta against a
`lot` floor is roughly ten times too permissive, and `size` is precisely where
the repository's largest surviving claims live — GroupDRO at −0.18 to −0.27.

**Fixed rather than annotated.** `scripts/determinism_repeats.sh` now takes a
protocol and encoder and writes `runs/determinism__<proto>__<enc>.json`;
`report.floors()` loads whatever has been measured and `report.floor_for()`
returns the protocol's own value, falling back to the **largest** measured
floor where a protocol has none — because being too strict withdraws a claim and
being too lenient publishes one, and only one of those errors is recoverable.
All three generators now use it, and the paper states plainly that the floor is
not one number.

`scripts/size_objectives_seeds.sh` measures the `size` floor first, before
running the 42-cell grid whose verdicts depend on it. That ordering matters: had
the grid run first I would have had 42 cells judged against a `lot` floor and a
strong temptation to keep the verdicts.

**The pattern worth naming.** Three times now the same mistake has appeared in a
different costume: seed spread standing in for run-to-run spread (§16), a
baseline from another batch standing in for a within-batch control (§18), and a
floor from another protocol standing in for this protocol's (here). Each time it
made an effect look more real than it was. The general form is *borrowing a
scale from a context that is not the one being measured*, and it is apparently
the single most reliable way to manufacture a finding in this repository.

### 29. Two pairs finally separate, and the credit belongs to seeds rather than to the session control

The `lot` cells of the single-session grid are in. First within-session
representation ranking with three seeds on every cell:

| representation | n | mean | range |
|---|---|---|---|
| cnn_gn | 3 | 0.8667 | [0.8597, 0.8742] |
| cnn_bn | 3 | 0.8543 | [0.8510, 0.8600] |
| feat | 3 | 0.8338 | [0.8283, 0.8417] |
| graph | 3 | 0.7524 | [0.7484, 0.7557] |

| adjacent pair | gap | verdict |
|---|---|---|
| cnn_gn > cnn_bn | +0.0124 | ranges overlap |
| cnn_bn > feat | +0.0205 | **separated**, margin 0.0093 > floor 0.0054 |
| feat > graph | +0.0815 | **separated**, margin 0.0727 |

Two adjacent pairs separate. Across the whole mixed-session table, **zero of
eighteen** did. That is the grid earning its keep — but not for the reason it
was launched. `cnn_bn > feat` was previously unresolvable because `feat` was
single-seed, not because its seeds were spread across sessions. The gain is from
having error bars at all, exactly as §27 found the session control itself does
nothing measurable. The script's header now says so.

### 30. The one comparison that keeps landing on the resolution limit

`cnn_gn` against `cnn_bn` has now been measured twice and landed on the boundary
both times, from opposite sides:

* mixed-session, three seeds: ranges **disjoint by 0.0004**, below the floor;
* one session, three seeds: ranges **overlap by 0.0003**.

Same verdict twice — not established — and the near-symmetry is what an effect
the size of the measurement noise looks like. It is worth settling rather than
leaving ambiguous, because "GroupNorm beats BatchNorm everywhere" is a named
claim with a real mechanism behind it: BatchNorm mixes statistics across
whatever is in the batch, and a batch here spans lots, so the choice is a
domain-leak question rather than a tuning one. A mechanism is a reason to expect
an effect, not evidence of one.

**And it could not have been settled at three seeds, for a reason that is
arithmetic rather than experimental.** Every verdict in this repository uses
"do the observed ranges overlap, and is the margin above the run-to-run floor".
That is the right conservative default for reading a table of many three-seed
cells, and it has one property that makes it the wrong tool here: **the range of
a sample grows with the sample size**, so running more seeds to answer a
question makes non-overlap strictly harder to achieve. A sweep would be
penalised for having more data.

So `scripts/gn_vs_bn.sh` runs both encoders at **eight seeds each, together**,
and `scripts/gn_vs_bn.py` reads it with an exact permutation test on the group
labels — assumption-free, and indifferent to sample size in the right
direction. The numbers that matter:

* eight per arm gives C(16,8) = 12,870 arrangements, so the smallest attainable
  two-sided p is 0.000155;
* **three per arm gives 20 arrangements, and the smallest attainable two-sided
  p is 0.100.**

That second figure is worth sitting with. Every three-seed comparison in this
repository, however carefully judged, was incapable of producing evidence
stronger than p = 0.1 even in the most favourable case where the arms separate
perfectly. The range criterion was not being conservative on top of a
well-powered design; it was the only honest thing available given the design.

Both readings are reported side by side, and if they disagree the disagreement
is the result. `tests/test_smoke.py` pins the permutation test against identical
arms, perfectly separated arms, arm-swapping, and the n=3 floor of 0.1.

**H30, recorded before the run:** the effect is real and about +0.012, and eight
seeds will give a permutation p below 0.05 while the range test still says
"overlap". If instead the permutation test also fails to separate, the effect is
smaller than this protocol can measure and the claim comes out of the README,
where it currently sits as a design principle.

### 31. The weekend's first withdrawal rests on an asymmetric comparison

The RPCA fourth-channel ablation (§1, §5) is the withdrawal I have quoted most,
and its conclusion — a channel of zeros buys what the decomposition buys — has
been load-bearing since Saturday morning. Checking the file timestamps, the
comparison is not symmetric:

| arm | seeds and when they were measured |
|---|---|
| `cnn_gn` 3-channel | s0 **08:54 Friday**, s1 11:01, s2 11:02 |
| RPCA residual | s0 **09:20 Friday**, s1 11:01, s2 11:02 |
| raw fail mask | s0 11:03, s1 11:04, s2 11:05 |
| zeros | s0 11:03, s1 11:04, s2 11:05 |

**Two arms carry a Friday cell and two do not.** Given the measured session
offset — §16 established that a Friday cell can sit outside all six repeats of
itself — that is a systematic difference between the treatment arm and its
controls, in a comparison whose whole purpose was to be symmetric.

Restricted to within-session cells only:

| arm | all seeds | within-session only | vs 3-channel (within-session) |
|---|---|---|---|
| `cnn_gn` 3-channel | 0.8647 (n=3) | 0.8635 (n=2) | — |
| RPCA residual | 0.8696 (n=3) | **0.8638 (n=2)** | **+0.0002** |
| raw fail mask | 0.8692 (n=3) | 0.8692 (n=3) | +0.0057 |
| zeros | 0.8689 (n=3) | 0.8689 (n=3) | +0.0054 |

The conclusion does not weaken; it sharpens, and in the direction least
flattering to the method. Within one session the RPCA residual buys **+0.0002**
over three channels while both controls buy about **+0.0055**, so the residual
sits 0.0051–0.0055 *below* its own controls — right at the floor. The reading is
no longer "the decomposition is worth what nothing is worth" but "a fourth
channel is worth a little, and the RPCA residual is the worst thing to put in
it".

**I am not adopting that reading yet, and the reason matters.** It rests on n=2
for the residual arm, and it was obtained by *selecting cells according to when
they ran*. That is a subgroup choice, and this repository has spent three
entries (§16, §18, §28) catching exactly this class of move — borrowing or
carving a comparison so that the scale suits the conclusion. Doing it in my own
favour would be no better for having a good reason.

So `scripts/rpca_ablation_clean.sh` runs **all four arms, three seeds each,
twelve cells, together**. No arm gets a cell the others do not. Queued last.

**H31, before the run:** the four arms come out within the floor of one another,
as the original ablation said, and the apparent −0.005 for the residual is n=2
noise. If instead the residual lands clearly below both controls at three clean
seeds, the finding upgrades from "the decomposition does nothing" to "the
decomposition actively costs something", and the most likely mechanism is that
the residual is a float channel with a different scale from the one-hot channels
it is concatenated to, which would be a normalization bug rather than a fact
about RPCA.

That last clause is a prediction I want on record, because it is the sort of
thing that is much easier to propose after seeing the number than before.

### 32. The mechanism I predicted is refuted; the one the data hands me is better

§31 recorded, before the clean ablation runs, that if the RPCA residual came out
below its controls the likely cause was a scale mismatch — a float channel
concatenated to one-hot channels. I said then it was the sort of explanation
much easier to propose after seeing a number than before, which is why it went
on record. It is wrong, and checking took one measurement of the tensor the
encoder actually receives:

| 4th channel | ch0 | ch1 | ch2 (fail one-hot) | ch3 |
|---|---|---|---|---|
| residual | 0.2096 / 0.4070 | 0.6613 / 0.4733 | 0.1291 / 0.3353 | **0.1291 / 0.3353** |
| failmask | 0.2096 / 0.4070 | 0.6613 / 0.4733 | 0.1291 / 0.3353 | **0.1291 / 0.3353** |
| zeros | — | — | 0.1291 / 0.3353 | 0.0000 / 0.0000 |

(mean / std). The residual channel is statistically indistinguishable from the
fail-mask channel *and* from the one-hot channel beside it. Only 5.21% of wafers
have a residual outside {0, 1} at all, and on those the values run
[−1.002, +1.026] with mean |v| 0.0770 — the same order of magnitude. **There is
no scale mismatch.** That explanation is now unavailable to me when the run
lands, which is the point of having written it down.

**What the data gives instead is a much better answer, and it inverts the whole
idea.** RPCA splits a lot into what its wafers share and what each does alone,
and hands the encoder the second on the theory that the shared part is the
tool's nuisance. That theory needs the defect to be per-wafer and the nuisance
to be lot-wide. Measured (`scripts/rpca_mechanism.py`):

| class | n | n decomposed | enrichment | failed-die mass removed |
|---|---|---|---|---|
| **Edge-Ring** | 9,680 | **7,519** | **15.77x** | **0.616** |
| Random | 866 | 192 | 4.50x | 0.524 |
| Donut | 555 | 119 | 4.35x | 0.388 |
| Near-full | 149 | 26 | 3.54x | 0.810 |
| Center | 4,294 | 174 | 0.82x | 0.467 |
| Edge-Loc | 5,189 | 140 | 0.55x | 0.364 |
| Loc | 3,593 | 80 | 0.45x | 0.236 |
| **Scratch** | 1,193 | **3** | **0.05x** | — |
| none | 147,429 | 268 | 0.04x | 0.004 |

The decomposition fires on 4.93% of wafers, and 7,519 of those 8,521 are
`Edge-Ring` — 77.7% of every `Edge-Ring` wafer in the corpus. On them it removes
**61.6% of the failed-die mass**.

`Edge-Ring` is edge roll-off. It is a lot-level process condition, shared across
the wafers of a lot *by definition*. So for the class RPCA touches most, **what
the lot shares is the defect**, and the low-rank part it subtracts as nuisance is
the label. The same holds for `Near-full` (0.810 removed), `Donut`, `Random` —
every pattern that recurs lot-wide.

And the converse is exactly as sharp: `Scratch` is enriched **0.05x** and `Loc`
0.45x. A scratch on one wafer is not shared by its lot, so the decomposition
never fires on it. Those two are the long tail this repository has spent the
weekend failing to move, and they are precisely the classes a lot-signature
method cannot reach even in principle.

**A per-lot low-rank prior is pointed the wrong way for this corpus.** It
subtracts signal from the classes that are lot-correlated, and offers nothing to
the classes that are not. That is a considerably better result than "the channel
is worth what zeros are worth" — it explains the 5% where the method does
something, rather than only the 95% where it does nothing, and it predicts
where the harm should show up.

**H32, recorded before the clean ablation lands:** if the residual arm comes out
below its controls, the deficit is concentrated in `Edge-Ring` per-class F1 and
is near zero for `Scratch` and `Loc`. If instead the deficit is spread evenly
across classes, this mechanism is wrong too and the difference is something
about the fourth channel in general rather than about what is in it.

The per-class F1 needed to check that is already recorded in every run JSON, so
the prediction can be scored the moment the twelve cells finish.

### 33. H32 fails its first test, and the failure explains the whole ablation

H32 predicted that any RPCA deficit would concentrate in `Edge-Ring`. The
`sess2` cells give an early read — `rpca_cnn` against `cnn_gn`, three seeds
each, one session — and the prediction is wrong:

| class | cnn_gn | rpca_cnn | delta |
|---|---|---|---|
| Near-full | 0.8792 | 0.9211 | **+0.0419** |
| Random | 0.8637 | 0.8810 | +0.0173 |
| Scratch | 0.7272 | 0.7418 | +0.0146 |
| **Edge-Ring** | 0.9836 | 0.9822 | **−0.0014** |
| Loc | 0.7765 | 0.7661 | −0.0104 |
| Center | 0.9300 | 0.9203 | −0.0096 |

`Edge-Ring` — the class the decomposition mutilates on 77.7% of its wafers,
removing 61.6% of the failed-die mass — is **the least affected class in the
table**. (This is a partial read: it compares `rpca_cnn` to `cnn_gn`, which
conflates "a fourth channel" with "what is in it". The clean residual-vs-controls
run will score H32 properly. But the direction is already clear.)

**And the reason is a line of code, not a property of RPCA.** `stack_channels`
concatenates the fourth channel to the **intact** one-hot:

```python
x = F.one_hot(maps64.long().clamp(0, 2), 3)...      # ch2 = raw failed dies
return torch.cat([x, resid.unsqueeze(1)], dim=1)     # ch3 = residual
```

Channel 2 still carries the raw failed-die mask. Whatever the decomposition does
to channel 3, the encoder has an untouched copy beside it and can ignore the
damage. Edge-Ring F1 is 0.9822 against 0.9836 because the model never needed
channel 3 at all.

The docstring says this was deliberate — *"handed to the encoder alongside the
raw state rather than instead of it"* — and as a safety property it works. But
**the safeguard that stops the decomposition hurting is the same thing that
stops it helping.** For 95% of wafers channel 3 is a bitwise duplicate of
channel 2; for the other 5% it is channel 2 with most of the defect deleted. It
has never been possible for that channel to carry information the encoder did
not already have. The ablation result — worth what a channel of zeros is worth —
was structurally guaranteed from the moment the concatenation was written, and
no number was needed to see it.

That is the strongest form of the finding and it is the one for the paper: not
"we ran a control and it tied", but "the control could not have failed to tie".

**H33, recorded before the run.** `--hide-raw-fail` zeros channel 2 for
`rpca_cnn`, removing the fallback so the fourth channel is the only description
of where the wafer failed. Two arms, three seeds, one session:

* `--sig-channel failmask --hide-raw-fail` — the true failed dies, once;
* `--sig-channel residual --hide-raw-fail` — the same, minus what the lot shares.

Prediction: with no fallback the residual falls clearly below the raw mask, and
the gap **does** concentrate in `Edge-Ring`, while `Scratch` (enrichment 0.05x)
and `Loc` (0.45x) are unaffected. If the gap is flat across classes instead, the
lot-shared-is-the-defect mechanism is wrong too and I am out of explanations
that the data supports.

This is the experiment that should have accompanied the original claim: it asks
what the decomposition does to the *representation*, rather than whether a model
that can route around it still scores the same.

### 34. Adjacent-pair testing is not stable under adding a competitor

Two turns ago I reported that the within-session `lot` grid resolved two adjacent
pairs where the mixed-session table had resolved none. `spectral` and `rpca_cnn`
have since landed and the count is back to **one**:

| representation | n | mean | range |
|---|---|---|---|
| rpca_cnn | 3 | 0.8717 | 0.0157 |
| cnn_gn | 3 | 0.8667 | 0.0145 |
| cnn_bn | 3 | 0.8543 | 0.0089 |
| spectral | 3 | 0.8405 | **0.0432** |
| feat | 3 | 0.8338 | 0.0134 |
| graph | 3 | 0.7524 | 0.0073 |

`cnn_bn > feat` was separated at margin 0.0093 when they were adjacent. They are
no longer adjacent: `spectral` sits between them, with the widest range of any
cell measured this weekend, so it overlaps both neighbours and both former
neighbours' comparison is no longer made.

Nothing changed about `cnn_bn` or `feat`. **The verdict moved because a third
cell arrived.** That is a defect in how I have been presenting the ranking:
adjacent-pair testing answers "is this cell distinguishable from the next one
down", which depends on what else is in the table, and it silently rewards a
sparse table. The claim "two pairs separate" was true of a four-row table and
false of the six-row table it was always going to become.

The right presentation is the full pairwise matrix, or comparisons against a
fixed reference. I am not going to rebuild the tables this late; instead
`RESULTS.md` keeps the adjacent-pair verdicts with this caveat attached, and the
durable statement — the one that does not depend on table composition — is that
**only the die-graph GNN separates from anything at all**, by 0.0815 against the
next-worst representation.

### 35. The forward-only drop, decomposed — and I had the second half backwards

The `lot_time` cells of the single-session grid are in, so the decomposition H12
asked for can finally be done end to end with every cell from one session at
three seeds.

| representation | `iid` all | `iid` @ the slice | `lot_time` | slice | everything else | total | slice share |
|---|---|---|---|---|---|---|---|
| CNN (GroupNorm) | 0.8837 | 0.8245 | 0.7002 | −0.0591 | −0.1244 | −0.1835 | **32%** |
| CNN (BatchNorm) | 0.8625 | 0.7903 | 0.6438 | −0.0721 | −0.1465 | −0.2186 | **33%** |
| descriptors + MLP | 0.8443 | 0.7617 | 0.6511 | −0.0826 | −0.1106 | −0.1931 | **43%** |

A third to nearly a half of the forward-only drop is present on a **random**
split with every one of those geometries seen in training and no temporal
structure whatever. The narrow slice is intrinsically hard. What is left —
−0.11 to −0.15 — is the part that deserves to be called drift, and it is still
the larger share.

**And the other half of my framing was wrong.** §8 and §23 described the drop as
"part temporal drift and part unseen geometry", on the strength of 14.15% of the
forward-only test wafers being of a geometry never trained on. Splitting
`lot_time`'s test set on that boundary, both halves averaged over only the eight
classes both contain:

| representation | seen geometry | unseen geometry | unseen − seen |
|---|---|---|---|
| CNN (GroupNorm) | 0.7032 | 0.7281 | **+0.0249** |
| CNN (BatchNorm) | 0.6560 | 0.6696 | **+0.0136** |
| descriptors + MLP | 0.6416 | 0.6387 | −0.0029 |
| die-graph GNN | 0.5909 | 0.5871 | −0.0037 |

**Unseen geometry is not the hard part.** On both CNNs it is *easier* than the
seen part of the same test set. So the component I flagged as a confound in the
headline does not push in the direction I assumed; it pushes weakly the other
way or not at all.

The corrected statement is two-part rather than three: the forward-only drop is
**a narrow, intrinsically hard geometry slice (a third to 43% of it) plus drift
and label shift (the rest)**, and meeting geometries you have never trained on
costs approximately nothing. That last clause is the interesting one for a fab —
it says the expensive thing about deployment is not novelty of product, it is
which products you happen to be running.

**What I did wrong, and it is the same shape as everything else this weekend.**
I inferred a direction from a *count* — 14.15% of test wafers are of unseen
geometry, therefore unseen geometry is part of the difficulty — without
measuring whether those wafers were actually harder. A count is not an effect.
The measurement was one field away in every run JSON from the moment
`geometry_decomposition` was written, and I quoted the count in three documents
across two turns before checking the thing it was standing in for.

Both corrections are now computed sections in `RESULTS.md` rather than prose, so
neither can drift back.

## Scoring the pre-registered hypotheses

Five hypotheses were written down before their runs. All five have now landed.
Two confirmed, two falsified, one void because the experiment did not do what I
thought it did.

### 36. H24 confirmed: pooling is the long tail, and it is the weekend's first positive result

`cnn_gn` on `lot`, three seeds each, one session, `lot` floor 0.0054:

| pool | macro-F1 | vs `mean` | Scratch | Loc | verdict |
|---|---|---|---|---|---|
| `mean` (original) | 0.8666 ±r 0.0128 | — | 0.7216 | 0.7714 | — |
| **`meanmax`** | **0.8839** ±r 0.0065 | **+0.0173** | **0.7782** | 0.7807 | **separated**, margin 0.0067 |
| `meanmean` (control) | 0.8717 ±r 0.0161 | +0.0052 | 0.7210 | 0.7744 | ranges overlap |

Per class against `mean`: `meanmax` moves **Scratch +0.0566, separated** at margin
0.0180; the capacity control moves it **−0.0006**. `Loc` +0.0093, overlapping.

**The control is what makes this a result.** `meanmean` has exactly the parameter
count of `meanmax` and carries the mean concatenated with itself. It buys
nothing. So the +0.0173 is max pooling, not a wider head — which is the question
that killed the RPCA channel and would have killed this one had it been the
answer.

The mechanism holds up: a `Scratch` is a thin connected line, global average
pooling reduces it to a mean over the wafer that is close to a slightly elevated
background failure rate, and keeping the extreme recovers it. Six weeks of
class-imbalance corrections did nothing to `Scratch`; changing what the encoder
pools moved it 0.057.

My prediction was *"Scratch and Loc move, macro-F1 barely does"*. Half right:
`Scratch` moved and separated, `Loc` did not, and macro-F1 moved **and**
separated — I understated the effect. Being wrong in the direction of the result
being larger than predicted is still being wrong.

### 37. H26 falsified: nothing separates from ERM on `size` either

The `size` protocol's own run-to-run floor, measured first: **0.0133**, two and a
half times `lot`'s. Then the 42-cell grid, three seeds, one session:

| objective | `cnn_bn` vs ERM | seed range | verdict | `feat` vs ERM | verdict |
|---|---|---|---|---|---|
| `group_dro` | **−0.1534** | 0.1844 | margin 0.0008 < floor | **−0.1341** | margin 0.0125 < floor |
| `logit_adjust` | −0.1117 | 0.1361 | margin 0.0018 < floor | −0.0756 | margin 0.0067 < floor |
| `irm` | −0.0258 | 0.1234 | overlaps | −0.0179 | overlaps |
| `coral` | −0.0200 | 0.0274 | overlaps | −0.0045 | overlaps |
| `dann` | −0.0171 | 0.0763 | overlaps | −0.0188 | overlaps |
| `mixup_domain` | −0.0112 | 0.0810 | overlaps | +0.0028 | overlaps |

ERM's own seed range on `size` is 0.0735 (`cnn_bn`) and 0.0603 (`feat`).

**I predicted GroupDRO would hold and it does not.** Its mean effect is −0.15,
the largest in the repository, and it still fails: its seeds span **0.1844**, so
its range and ERM's overlap almost exactly and the margin is 0.0008. GroupDRO on
`size` is not a method that loses; it is a method that is *unstable*, sometimes
catastrophically and sometimes not.

The error in my prediction is the fourth instance of the same mistake. I compared
−0.27 against "the seed half-range on `size`", meaning ±0.035, measured on
`cnn_gn` under ERM. The actual seed range for the cell in question is five times
that. I have now made this error with seed-vs-run-to-run spread (§16), a
cross-batch baseline (§18), a cross-protocol floor (§28), and now a
cross-*objective* seed range — and each time the effect looked realer than it
was.

**The domain-generalization result is now uniform.** Across `lot` and `size`,
under a degenerate domain definition and a real one, at three seeds with each
protocol's own floor: **not one of seven borrowed objectives separates from ERM
anywhere.** That is duller than "GroupDRO is catastrophic on geometry shift" and
it is what the data supports.

### 38. H30 confirmed, exactly, including the part where the two criteria disagree

GroupNorm against BatchNorm on `lot`, eight seeds per arm, one session:

* mean difference **+0.0130** — predicted "about +0.012";
* exact permutation test, 12,870 arrangements: **p = 0.0106** — predicted "below 0.05";
* range test: **ranges overlap** — predicted "still says overlap".

All three parts of the prediction hold, and the disagreement between the two
criteria is the methodological result it was meant to be. The range test cannot
resolve this and never could: it grows stricter with sample size, so the eight
seeds that settle the question make non-overlap *harder*, and at three seeds per
arm the smallest attainable two-sided p is 0.1 regardless of the data.

So **GroupNorm does beat BatchNorm on `lot`, by 0.0130, p = 0.011.** The claim
in `README.md` survives, and it is the only one of this repository's original
claims that has survived the weekend intact. It survives because it was
re-measured with a test capable of resolving it, not because it was believed.

### 39. H31 confirmed: the clean RPCA ablation says what the original said

All four arms, three seeds, one session, no arm holding a cell the others do not:

| arm | macro-F1 | vs 3-channel | verdict |
|---|---|---|---|
| `cnn_gn` 3-channel | 0.8703 ±r 0.0192 | — | — |
| RPCA residual | 0.8701 ±r 0.0195 | −0.0002 | ranges overlap |
| raw fail mask | 0.8712 ±r 0.0113 | +0.0009 | ranges overlap |
| zeros | 0.8669 ±r 0.0130 | −0.0035 | ranges overlap |

Residual vs raw mask −0.0011, residual vs zeros +0.0033, both overlapping. The
apparent −0.005 deficit from §31's within-session subgroup was n = 2 noise,
exactly as predicted, and **declining to adopt that reading was the right call**.
It was the most tempting number of the weekend — it made my own withdrawal look
sharper — and it was not real.

### 40. H33 void: the experiment did not do what I designed it to do

`--hide-raw-fail` zeroed the one-hot's fail plane so that the fourth channel
would be the only description of where the wafer failed. It ran, and returned a
clean null: residual vs raw mask −0.0020, ranges overlap, `Edge-Ring` among the
least affected classes at −0.0014.

That null is meaningless. The one-hot is over {outside, pass, fail} and **its
three planes sum to 1 everywhere**, so zeroing the fail plane leaves it exactly
recoverable as `1 − ch0 − ch1` — a single 1×1 convolution, which the first conv
layer is. Verified: the reconstruction error is 0.00e+00. Nothing was hidden and
the model never lost anything.

I would have accepted this null. It agreed with H31, it agreed with the early
read in §33, and it was the third piece of evidence pointing the same way. What
made me check was that it disagreed with a *mechanism* I had measured directly —
RPCA removes 61.6% of `Edge-Ring`'s failed-die mass, and a model given only that
should notice. When a well-measured mechanism predicts an effect and a clean
experiment says zero, the experiment is the thing to doubt first.

The six cells are renamed `*.void.bak` rather than deleted, so the record of the
mistake survives. The fix collapses pass and fail into a single "inside the
wafer" plane, so the partition exists only in the fourth channel, and
`tests/test_smoke.py` now asserts that no linear combination of the visible
planes equals the fail mask. Re-running now.

**The general lesson, and it is the sharpest of the weekend.** Every other error
here was a scale borrowed from the wrong context. This one is different and
worse: an experiment that could not have produced a positive result, returning a
negative one that agreed with everything else I believed. A control that cannot
fail is not a control, and a null from an experiment with no power is not
evidence — it is silence.

### 41. H33 re-run with power: the effect is real, my mechanism is not

The void experiment is redone with the fail plane genuinely hidden — pass and
fail collapsed into one "inside the wafer" plane, so the partition exists only
in the fourth channel.

| fourth channel | macro-F1 | seeds | verdict |
|---|---|---|---|
| raw fail mask only | 0.8705 ±r 0.0071 | 0.8680, 0.8684, 0.8750 | — |
| **RPCA residual only** | **0.8434** ±r 0.0178 | 0.8336, 0.8450, 0.8515 | **−0.0271, separated** (margin 0.0165 > floor 0.0054) |

And the design check that the void version failed: with the fail plane *visible*
the same two arms are 0.8712 and 0.8701, indistinguishable. Hide it and they
separate by 0.027. So the fallback was exactly what masked the difference, the
first null was a power failure, and **the RPCA residual is a materially worse
description of where a wafer failed than the raw mask** — 0.027 worse, when it
is the only description available.

That vindicates the mechanism at the level it was measured: the decomposition
does damage the representation. It also means the original ablation's conclusion
was right for a reason it never tested — the fourth channel is worth nothing not
because the residual is harmless but because the encoder was handed an
untouched copy of what the residual destroys.

**My per-class prediction is falsified, and cleanly.** I predicted the damage
would concentrate in `Edge-Ring`, the class RPCA fires on 77.7% of the time and
strips 61.6% of the failed-die mass from:

| class | baseline | mass removed | enrichment | damage |
|---|---|---|---|---|
| **Donut** | 0.7943 | 0.388 | 4.35x | **−0.1213** (separated) |
| Near-full | 0.9188 | 0.810 | 3.54x | −0.0649 |
| Random | 0.8757 | 0.524 | 4.50x | −0.0270 |
| Center | 0.9308 | 0.467 | 0.82x | −0.0172 |
| Loc | 0.7740 | 0.236 | 0.45x | −0.0168 |
| **Edge-Ring** | 0.9841 | **0.616** | **15.77x** | **−0.0038** (overlaps) |
| Scratch | 0.6996 | — | 0.05x | +0.0221 |

`Edge-Ring` has the most mass removed of any class and is the **second least
damaged**. `Donut` has less than two thirds as much removed and loses 0.12.

The obvious reading is that damage follows how much *margin* a class had rather
than how much mass was taken: `Edge-Ring` sits at 0.984 and a partial ring is
still unmistakably a ring, while `Donut` sits at 0.794, is 555 wafers, and is
distinguished from `Center` by a hole that deleting a third of the mass can
close. I cannot support that over the simpler story with these data — across the
eight classes with a measurement, Spearman(damage, baseline F1) is +0.52 and
Spearman(damage, mass removed) is −0.45, both weak, both in the expected
direction, and eight points cannot separate them. Reported as descriptive.

What survives without qualification: **removing "what the lot shares" costs
0.027 of macro-F1 when nothing else describes the defect, and the cost lands
almost entirely on classes other than the one the decomposition targets.**

### 42. The one positive result exists on one protocol, which is not enough here

`meanmax` is the only thing this weekend found that works. It is measured on
`lot`, with `cnn_gn`, and it would be quoted as "the long tail is an
architecture problem".

A repository whose entire thesis is *results move when the protocol changes* has
no business reporting a single-protocol win. `scripts/pooling_protocols.sh` runs
`iid`, `size` and `lot_time` at three seeds for all three variants including the
capacity control — 27 cells, launched.

**H42, before the run.** On `iid` the effect holds or grows, because max pooling
is a strictly richer statistic and the test distribution matches training. On
`lot_time` I expect it to **shrink or reverse**: max pooling keeps the extreme a
thin structure produces, and under forward-only shift the extremes are exactly
what a new tool or product changes, so the mean may be the more robust statistic.
If it reverses, the claim becomes "use max pooling when your test set resembles
your training set", which is materially weaker and is the version the data would
support.

I have been wrong about the sign of a per-class effect once already today, so
this is a prediction about direction only, and the control decides whether any
of it counts.

### 43. The Monday document said the weekend produced no positive results, which stopped being true

`WEEKEND.md` opened with *"the weekend produced essentially no positive results,
and that is the honest outcome"*. That was accurate when written and is now
false: `meanmax` beats global average pooling by +0.0173 macro-F1 and +0.0566 on
`Scratch`, both clearing the floor, with a capacity control that buys nothing.
It is the single actionable finding here and it was **absent from the Monday
document entirely** while five paragraphs of withdrawals were not.

That is a failure mode worth naming, because it is the mirror image of the one
this log has spent thirty entries on. Having spent the weekend learning to
distrust positive results, I wrote a hand-off that under-reported the one
positive result that had survived a control designed to kill it. Scepticism
applied asymmetrically is not scepticism, it is a different bias with better
manners.

Rewritten. The document now opens with the floor, states that most of what
follows is a withdrawal, and then says plainly that two things came out positive
and both were controlled. `2.0` is the pooling result, placed first in section 2
because it is the only thing an owner can act on.

Three further staleness fixes in the same pass:

* **§2.1 said the `size` half of the DG result "stands".** It does not. The
  re-run at three seeds with `size`'s own floor (0.0133) shows GroupDRO's
  −0.1534 failing at margin 0.0008, because its seeds span 0.1844 — more than
  its own effect. The section now carries that table and the conclusion is
  stated without an exception: no borrowed objective separates from ERM on
  either protocol, under either domain definition.
* **"What survives the floor"** was four rows and is now seven, gaining the two
  positives and the corrected H33 result.
* **The stage table** listed six stages as pending questions when five had been
  answered. Each now carries its answer, including the `rpca_hide_raw` row
  marked *"answered on the second attempt; the first was void"* — a hand-off
  that hides its own void experiment is not a hand-off.

The document is 3,213 words, up from 2,508. That is over the five-minute target
and I am accepting it: the previous version hit the target partly by omitting a
result. If it has to be one or the other, a tired person is better served by a
complete document they skim than a short one that misleads.

**One open question remains**, and it is the right one to be running into
Monday: the pooling win is measured on `lot` with one encoder, and a repository
whose thesis is that results move when the protocol changes has no business
reporting a single-protocol win. 27 cells across `iid`, `size` and `lot_time`
are in flight with the prediction recorded — holds or grows on `iid`, shrinks or
reverses on `lot_time`.

### 44. H42's first half is falsified, in the direction that makes the result more interesting

The `iid` cells of the protocol sweep are in. My prediction was that the pooling
effect would *hold or grow* on `iid`, because max pooling is a strictly richer
statistic and the test distribution matches training.

| protocol | `meanmax` vs `mean` | verdict | Scratch | verdict |
|---|---|---|---|---|
| `lot` | **+0.0173** | clears the floor | **+0.0566** | clears the floor |
| `iid` | +0.0090 | below the floor (0.0041) | +0.0253 | overlaps |

It shrinks, and it stops clearing the floor. The control behaves in both cases
(`iid` −0.0003, `lot` +0.0052, both overlapping), so this is not the control
failing — it is the treatment being genuinely smaller where there is no shift.

**The benefit is larger under lot-disjoint shift than without it.** That is the
reverse of the reasoning I recorded, and it points somewhere more interesting
than "a richer statistic is better": max pooling appears to be buying
*generalization* rather than capacity. A mean over the wafer is a statistic that
a new lot can move — change the background failure rate and the mean of every
class moves with it — while the max of a thin structure is comparatively
invariant to that. If that is what is happening, the effect should be largest on
the protocols with the most shift.

Which means **my prediction for `lot_time` is probably wrong too, and in the
same direction.** I said max pooling would shrink or reverse there, because
forward-only shift is where extremes move. The `iid` result argues the opposite:
if the benefit grows with shift, `lot_time` should show the largest effect of
the four. I am recording that revision now, before those cells land, so that
whichever way it goes it is scored against a prediction made in advance rather
than a story assembled afterwards.

I want to be careful about one alternative reading. `iid`'s baseline is 0.8836
against `lot`'s 0.8666, so there is less headroom and a ceiling effect could
produce a smaller absolute gain without any generalization story. Two things
would distinguish them: a ceiling effect should compress `Scratch` (0.72
baseline, far from ceiling) much less than macro-F1, and it does not — `Scratch`
also drops from +0.0566 to +0.0253. And if the shift story is right, `size` and
`lot_time`, whose baselines are *lower* than `lot`'s, should show effects at
least as large. Both are testable when the sweep finishes and neither requires a
new run.

### 45. The paper had the same omission the hand-off did

`paper_draft.md` did not contain the pooling result. Its only mentions of
"pooling" were in §7.2, listing it as an untried lever — written before the
experiment existed and never updated once it did.

This is the second document in two turns found to be missing the weekend's only
positive finding, and the two omissions have the same cause: every generated
section was written in the turn that produced its result, and a result that
arrived after a section was written does not retroactively appear in it. The
generators regenerate *numbers* faithfully and do not regenerate *structure* at
all. That is a real limitation of the "prose in the script, numbers from JSON"
design, and it is worth stating in the hand-off rather than leaving for someone
to trip over: **the tables cannot go stale; the section list can.**

Added as §4.3, placed immediately after the two contributions that failed their
controls, which is where it reads best — here is what did not work, here is what
did, and here is the control that makes the difference between the two claims.
It carries the `iid` result and states the claim at the strength the completed
protocols support rather than at the strength `lot` alone would allow.

### 46. A coverage check, and the one orphan it found closes the RPCA story

§45 named a structural limitation: the generators regenerate numbers faithfully
and do not regenerate structure, so a result can exist that no document reports.
`scripts/coverage_check.py` enumerates the experiment families in `runs/` and
checks each is consumed by a generator.

**The first version reported 24 orphans and nearly all were false.** It searched
the *documents* for each file's name — but a document discusses MIXED-SYNTH at
length without ever writing the string `mixedsynth__iid__bce__s0`. The question
that matters is whether any generator *reads* the result, since nothing
unreadable can be reported. Rewritten that way it finds **one**:
`runs/rpca_lambda.json`.

Worth noting that I caught this before quoting the 24. A check that cries wolf
is worse than no check, and I have spent the weekend on the harm done by numbers
that mean something other than they appear to.

**The orphan turned out to be the last piece of the RPCA story.** It is the
sweep behind the decomposition's one hyperparameter, and its docstring said the
choice was "documented rather than asserted". The documentation is measured on
the wrong population.

The script takes `[:40]` of the lots with at least 20 wafers, sorted by lot id.
Reading the shipped decomposition's own signature column:

| sample | mean rank at the standard weight | fraction rank 0 |
|---|---|---|
| **first 40 by lot id** (what the sweep used) | **0.475** | 0.525 |
| random 40, three seeds | 0.050, 0.025, 0.050 | 0.950–0.975 |
| all 6,504 lots | 0.050 | 0.951 |

The first-40 figure reproduces the sweep's reported 0.475 exactly, which
confirms I have the selection right. It is roughly ten times more likely to
yield a non-trivial decomposition than a random sample. Lot id is not arbitrary
here — §8 measured geometry and failed-die rate both drifting with it — so
`[:40]` of a sorted list is a slice, not a sample, and this was the most
favourable slice available. My first guess, that the discrepancy was lot size
(the sweep required ≥20 wafers, production uses ≥12), is refuted: rank 0 for
94.99% of ≥20-wafer lots against 94.83% of ≥12.

**Re-run on 200 random lots, the question the sweep should have answered:**

| lambda | mean rank | sparse share | low-rank energy |
|---|---|---|---|
| x0.5 | 0.00 | 1.000 | 0.000 |
| **x1.0 (standard)** | **0.04** | 0.978 | **0.022** |
| x1.5 | 0.11 | 0.967 | 0.032 |
| x2.0 | 0.32 | 0.950 | 0.049 |
| **x3.0** | **10.23** | 0.596 | **0.382** |
| x4.0 | 17.79 | 0.258 | 0.727 |
| x8.0 | 24.02 | 0.005 | 0.994 |

**There is no useful operating point.** Up to twice the standard weight the
low-rank part is empty and the decomposition returns its input; one step further
and the rank jumps from 0.32 to 10.23 and it absorbs 38% of the failure energy,
which on this corpus means it has begun eating the defect (§32: the class it
fires on hardest is `Edge-Ring`, whose defect *is* what the lot shares). The
transition is abrupt and nothing sits between the two failure modes.

So the RPCA account is now complete at every level, and each level was measured
rather than argued:

1. **The hyperparameter** has no good setting on a representative sample.
2. **The decomposition** therefore returns rank 0 for 95% of wafers, and where
   it does fire it removes 61.6% of `Edge-Ring`'s failed-die mass.
3. **The residual** is consequently a materially worse description of the defect
   than the raw mask — 0.027 worse when it is the only description available.
4. **The channel** is nonetheless worth exactly what a channel of zeros is
   worth, because `stack_channels` hands the encoder an untouched copy of what
   the residual destroys.

Point 4 was the original finding and it is the least interesting of the four. It
took until the last orphan to see that the method never had a setting at which
it could have worked, and the evidence that said otherwise was forty lots chosen
by a slice.

### 47. H42 answered: the weekend's one positive result holds on one protocol out of four

| protocol | baseline | `meanmax` | verdict | Scratch | verdict | control |
|---|---|---|---|---|---|---|
| `iid` | 0.8836 | +0.0090 | below floor | +0.0253 | overlaps | +0.0003 |
| **`lot`** | 0.8666 | **+0.0173** | **separated** | **+0.0566** | **separated** | +0.0052 |
| `lot_time` | 0.7046 | +0.0051 | overlaps | +0.0291 | overlaps | +0.0169 |
| `size` | 0.8404 | **−0.0590** | below floor | −0.0777 | below floor | +0.0022 |

The capacity control overlaps on every protocol, so the instrument is clean
throughout; what varies is the treatment.

**Both of my predictions were wrong, and they were wrong in opposite
directions.** H42 said the effect would hold or grow on `iid` because a richer
statistic should help most where train and test match — it is *smaller* there
than on `lot`. Then, after seeing `iid`, I revised to "the benefit grows with
shift, so `lot_time` should be largest" — and `size`, the protocol with the most
shift of the four, is the one place the effect is negative. Two hypotheses, both
recorded in advance, both falsified by the data they were written for. That is
the system working, and it is worth noting that neither would have been
falsifiable if I had waited to write them until after the cells landed.

What the four protocols line up with is **how much geometry the test set shares
with training** — 99.95%, 99.7%, 86%, 0% against +0.009, +0.017, +0.005, −0.059.
A max over a resampled feature map depends on the die density of the wafer it
came from, and an unseen geometry resamples differently, so the statistic that
helps when geometry is shared may be the one that breaks when it is not. **That
is a story assembled after the fact from four points and I am labelling it a
hypothesis in the paper, not a result.** I have spent this weekend documenting
what happens when a plausible mechanism is stated as though it were measured.

**The claim that survives is narrow and it is the right size.** On a
lot-disjoint split, mean-and-max pooling moves the hardest class by 0.057 while
its capacity control moves it by −0.001. On the other three protocols it is not
established, and on geometry holdout its mean is negative. An architectural
change that read as a general improvement is protocol-specific — which is
precisely what this benchmark was built to detect, turned for once on our own
result rather than on a borrowed method.

**One loose end, and it is the largest apparent effect in the sweep.** The
`size` figure of −0.0590 does not clear that protocol's floor: the margin is
0.0068 against 0.0133, because `size` seed ranges are enormous. So the biggest
number in the table is also the least established, which is the shape of half
the errors in this log. Three seeds cannot settle it — an exact permutation test
with three per arm has 20 arrangements and a smallest attainable two-sided p of
0.1, regardless of the data. `scripts/pooling_size_seeds.sh` takes both arms to
eight seeds and reads them with the permutation test, the same instrument that
resolved GroupNorm against BatchNorm when the range test could not.

**H47, before the run:** the −0.059 is real and eight seeds give a permutation
p below 0.05. If it does not, max pooling is simply unestablished either way on
`size`, and the honest summary of the weekend's one positive finding is that it
holds on one protocol out of four and is unmeasured on the rest.

### 48. The fourth silent no-op, caught before it produced a missing answer

`scripts/pooling_size_seeds.sh` ends by invoking the permutation test to resolve
the `size` pooling effect. The invocation cannot work. `gn_vs_bn.py` globs

```
runs/lot__cnn_*__erm__{tag}__s*.json
```

with the protocol, both encoders and the objective hardcoded, because it was
written for exactly one comparison. Asked for `--tag poolsize` on the `size`
protocol it matches **zero files**, prints `need equal, >=2 seeds per arm; have
0 and 0`, and returns 0. The stage would have completed, logged success, and
produced no `runs/pooling_size_perm.json` — and I would have gone looking for
the answer to H47 and found nothing, with no error to explain why.

That is the fourth silent no-op of this project, after the redirect into a
nonexistent directory that dropped three cells, the reporter key collision that
would have overwritten seeds, and the OOM that killed the first wafer-budget run
while its stage logged `done`. All four share the property that the harness
reported success. The pattern is now frequent enough to be the finding rather
than the anecdote: **in this codebase, the default failure mode is silence, and
every stage boundary is a place where an error can be converted into an absence.**

Three things done about this one:

* **The tool is generalized.** `--protocol`, `--objective`, `--arm-a`, `--arm-b`
  make it a permutation test between any two cell families, which is what it
  should have been when it was written. The legacy path is unchanged and
  verified: re-running the GroupNorm/BatchNorm comparison reproduces p = 0.01057
  and the same difference to five decimals.
* **An empty match is now an error.** It prints which arm had how many seeds and
  exits 2, so a queued caller cannot mistake "matched nothing" for "nothing to
  report", and it writes no output file rather than an empty one.
* **The contract is asserted** in `tests/test_smoke.py`: an empty match must
  return 2, print `ERROR`, and leave no file behind.

I did not edit `pooling_size_seeds.sh`, which was executing at the time. Bash
reads a script by byte offset and rewriting one mid-run can drop it into the
middle of a line — I did that once already this weekend with `chain_rest.sh` and
had to kill and relaunch it. The corrected call is queued as a separate stage
that waits for the sweep's own completion marker.

**What made me look.** Nothing failed. I checked because the previous turn ended
by saying the analysis was queued, and writing that sentence prompted the
question of whether the thing I had queued was the thing I needed. Of the four
silent no-ops, three were found by that kind of incidental check and one by the
absence of an output file. None was found by anything failing, because nothing
did.

### 49. A counter for the absences

§48 named the pattern; this turn builds the guard. `scripts/verify_stage.py`
takes a glob and an integer and fails loudly when a stage produced fewer files
than it launched. Demonstrated on both a stage that worked and the one that did
not:

```
$ verify_stage.py --glob 'runs/lot__cnn_gn__erm__poolmeanmax__s*.json' --expect 3
  ok      3 / 3   ...   exit 0

$ verify_stage.py --glob 'runs/pooling_size_perm.json' --expect 1
  BAD     0 / 1   ...
  FAILED: size permutation test did not produce what it launched.   exit 1
```

Wired into the three stages that end by writing a summary file — `gn_vs_bn.sh`,
`al_wafer_budget.sh`, `determinism_repeats.sh` — because that is where three of
the four silent failures landed. `pooling_size_seeds.sh` was executing and was
not touched, for the byte-offset reason.

**Why a glob and an integer rather than something better.** A more principled
guard would reconstruct each launched cell's expected filename from its
arguments and check them individually. `run_bench.py` already owns that mapping
(`{protocol}__{encoder}__{objective}{suffix}__s{seed}.json`) and duplicating it
in bash would be a second place for it to drift. The dumb version catches every
failure this project has actually had, costs one line per stage, and will still
be added by someone in a hurry. A check that is hard to add does not get added,
and the failure mode here is not subtlety — it is nothing happening.

**What it does not catch, stated so nobody trusts it further than it goes.** It
counts files, so it cannot see a stage that wrote the right number of wrong
results. The reporter key collision (#2 in the list) would have produced exactly
the right file count while silently overwriting a seed; it was caught by reading
the code, not by counting. Overwrites, wrong-population samples like the `[:40]`
in the lambda sweep, and controls that cannot fail are all invisible to it. It
raises the floor and does not raise the ceiling.

`WEEKEND.md` now carries the four failures as a table under "what is running",
with the corollary that matters for anyone reading the results: **a missing
result is not a null result.** Where a table says `[not measured]`, check
whether the run happened before concluding the effect is zero. That sentence is
there because I very nearly made that mistake myself with H33 — a void
experiment returned a clean null that agreed with everything I already believed,
and I would have kept it if it had not contradicted a mechanism I had measured
directly.

### 50. H47 falsified: the `size` pooling effect halves and does not survive

Eight seeds per arm, one session, read with the exact permutation test:

| arm | seeds | mean |
|---|---|---|
| `meanmax` | 0.8039, 0.7357, 0.8046, 0.8076, 0.8758, 0.8191, 0.8601, 0.8383 | 0.8181 |
| `mean` | 0.8231, 0.8114, 0.8868, 0.8088, 0.8756, 0.8202, 0.8871, 0.8568 | 0.8462 |

Difference **−0.0281**, ranges overlap, permutation **p = 0.162** over 12,870
arrangements.

I predicted the −0.0590 was real and would give p < 0.05. It does not, and the
effect **halved** as seeds were added: −0.0590 at n=3, −0.0281 at n=8. The
three-seed estimate was off by more than the eight-seed effect is large. So max
pooling on the geometry-holdout protocol is **not established either way**, and
the fallback position I wrote into the script before running it is the one that
stands: the weekend's positive finding holds on one protocol out of four and is
unmeasured on the rest.

Worth being explicit that this is the third time a striking three-seed number
has shrunk or vanished under more seeds — GroupDRO's −0.27 on `size`, the RPCA
residual's −0.005 within-session subgroup, and now this. Each time the direction
survived and the magnitude did not, and each time the three-seed version was the
one that would have been quotable.

### 51. Applying my own standard to my own positive result

There is an asymmetry in what I have just done and it needs fixing rather than
noting.

I took the `size` effect to eight seeds because I suspected it was noise, and it
was. The `lot` effect — `meanmax` +0.0173 macro-F1 and +0.0566 on `Scratch`,
the one positive result of the weekend — is still at **three seeds**, the sample
size that has been wrong three times, and I have been quoting it in `WEEKEND.md`
and `paper_draft.md` as the headline finding.

Demanding eight seeds of a result I expected to be negative while accepting
three for one I expected to be positive is not scepticism, it is a preference
wearing scepticism's clothes. §43 caught the same asymmetry running the other
way — a hand-off that under-reported the positive result because I had spent the
weekend learning to distrust positives. Both are the same error: letting what I
expect decide what standard applies.

`scripts/pooling_lot_seeds.sh` takes both arms on `lot` to eight seeds and reads
them with the same permutation test, and ends with `verify_stage.py` on both the
cells and the summary, so it cannot log completion having produced nothing.

**H50, recorded before the run.** The `lot` effect survives with a permutation
p below 0.05 on macro-F1, and a smaller p on `Scratch` — because `Scratch` is
where the mechanism says the effect should live, and its three-seed margin
(0.0180) was nearly three times the floor rather than barely over it, unlike the
macro-F1 margin (0.0067) which cleared 0.0054 by a hair.

If it does not survive, the weekend produced no positive result at all, and
`WEEKEND.md` goes back to the sentence I removed from it two turns ago. I would
rather find that out now than have it found in a review.

### 52. The instrument could not reach half of the prediction it was meant to score

H50 has two parts: the `lot` pooling effect survives eight seeds with a
permutation p below 0.05 on macro-F1, **and** a smaller p on `Scratch`, because
`Scratch` is where the mechanism says the effect lives and its three-seed margin
was nearly three times the floor while macro-F1's cleared by a hair.

The permutation tool reads `test.macro_f1` and nothing else. The sweep I
launched last turn calls it once, on macro-F1. So the `Scratch` half of the
prediction would have arrived unscoreable, and the likely outcome is obvious:
report the macro-F1 result, quietly not mention the other half, and nobody
including me notices that a prediction was made and then half-abandoned because
the tool was inconvenient.

That is a small version of a failure this log has documented at larger scale
four times — an absence produced at a boundary, invisible because nothing fails.
Here the boundary is between what I predicted and what the instrument can
measure.

`--metric` now takes `macro_f1` or `class:<name>`. The legacy path is verified
unchanged (GroupNorm vs BatchNorm reproduces p = 0.01057 and the same difference
to six decimals), a test pins that the tool follows the metric it is asked for
rather than always reading macro-F1, and `chain_s15.sh` runs the `Scratch`
comparison when the sweep finishes, behind `verify_stage.py`.

**A number that fell out of building it.** The `size` sweep, re-read on
`Scratch` at eight seeds: `meanmax` 0.6660 against `mean` 0.7087, difference
−0.0427, **p = 0.202**. So `size` is unestablished on *both* metrics, not just
macro-F1, which makes §50's conclusion firmer than it was — the three-seed
`Scratch` figure of −0.0777 was as much an overestimate as the macro-F1 one.

Two things about that worth saying plainly. First, it is a number I would not
have had if the tool had stayed single-metric, and I only got it because
building the option let me point it at existing data for free. Second, the
direction is stable across metrics and sample sizes while the magnitude is not,
which is now the fourth instance of the same pattern and probably the most
useful single generalization this weekend produced: **on this corpus, three
seeds are enough to get the sign right and not nearly enough to get the size
right.**

### 53. A document regenerated during a sweep is current and partial at the same time

Rewriting the abstract exposed a property of the "prose in the script, numbers
from JSON" design that I had not thought about. The generators read `runs/` at
the moment they are invoked. If a sweep is in flight, they read *some* of it.

The pooling figure moved while I was editing the paragraph that quotes it:
+0.0173 at three seeds when the sweep was queued, +0.0158 partway through,
+0.0154 at seven of eight. Every one of those is honestly computed from cells
that exist, and every one is a different number in the abstract of a document
that claims to be regenerated rather than typed.

Nothing here is wrong. That is what makes it worth writing down: the guarantee
the design offers is "no number was typed by hand", which is not the same as
"this number is the one the experiment will produce". A reader who diffs two
regenerations of `paper_draft.md` an hour apart will find figures that moved
with no commit explaining why, and the explanation is that the experiment was
still running.

The fix is small and general: **print the sample size next to the number.**
`fmt()` already did for the stat helper; the pooling tables in all three
generators did not, and now do. A reader who sees `+0.0154 (n=7 vs 7)` against a
sweep specified as eight per arm can tell at a glance that they are looking at a
partial result, and a reader who sees `(n=3)` next to the capacity control can
tell it was not extended.

Worth connecting to §49: `verify_stage.py` counts files at the *end* of a stage
to catch absences. This is the same problem sampled at a different time — the
count is right eventually, and any document generated before "eventually" is
partial without saying so. Both are instances of a general shape: **in a system
where reports are derived continuously from an evolving directory, "regenerated
from data" is a claim about provenance and not about completeness.**

For the hand-off: if a figure in `RESULTS.md`, `paper_draft.md` or `WEEKEND.md`
disagrees with one in `critique_log.md`, check the `n`. The critique log records
what was measured at the moment it was written and is deliberately not
regenerated; the other three are always current, which during a sweep means
always partial.

**Addendum, one regeneration later.** At seven seeds per arm the `lot` pooling
comparison now reads *ranges overlap* on both macro-F1 and `Scratch`, where at
three it read *separated* on both. Nothing about the effect changed direction —
+0.0154 macro-F1 and +0.0488 `Scratch`, against +0.0173 and +0.0566 at n=3 — but
the range criterion has stopped resolving it, exactly as the arithmetic said it
would: a sample's range grows with the sample, so adding seeds makes non-overlap
strictly harder to achieve.

This is the situation the permutation test was built for, and the fact that it
arrived on the weekend's one positive result rather than on a convenient
negative is the best available check that the instrument was not chosen to suit
a conclusion. The verdict on H50 will come from a test that does not punish
extra data, and I will take whatever it says.

### 54. H50 confirmed on both counts: the weekend has a positive result

Eight seeds per arm on `lot`, one session, exact permutation test:

| metric | `meanmax` | `mean` | difference | range test | permutation p |
|---|---|---|---|---|---|
| macro-F1 | 0.8888 | 0.8738 | **+0.0150** | ranges overlap | **0.01197** |
| **`Scratch`** | 0.7764 | 0.7292 | **+0.0473** | ranges overlap | **0.00047** |

The prediction was: survives with p < 0.05 on macro-F1, and a *smaller* p on
`Scratch`, because `Scratch` is where the mechanism says the effect lives. Both
hold, and `Scratch`'s p is twenty-five times smaller than macro-F1's.

**That concentration is the part that matters.** The claim was never "a bigger
head helps"; it was that a global average destroys the one thing that identifies
a thin connected line, and a max preserves it. If the effect were capacity or
noise it would be spread across the nine classes, and it is not — it is
concentrated in exactly the class the mechanism named, and the capacity control
with identical parameter count moves that class by −0.0056.

**And the range criterion says "ranges overlap" for a result at p = 0.00047.**
That is the clearest demonstration this weekend of the point made in §30: the
range of a sample grows with the sample, so the criterion that is right for
reading a table of three-seed cells becomes actively misleading once a
comparison has been run properly. `RESULTS.md` now prints the permutation test
beside the range verdict with an instruction to prefer it, because a table
saying "overlaps" next to p = 0.0005 is worse than no table.

So the weekend produced one positive result, and it is now held to a higher
standard than anything else in this repository: eight seeds per arm, an exact
test that does not punish extra data, a capacity control that moves nothing, a
mechanism that predicted *which class* would move before the run, and a known
failure to generalize.

**What it is worth being careful about.** It is one encoder, one protocol, and a
first-order architectural change that anyone could have tried. It does not
rescue the benchmark's headline numbers, it does not approach the 0.95 target,
and its own protocol-dependence is the more interesting half. I am recording it
as a positive result and not as a contribution.

### 55. The other half of that claim is still at three seeds

The finding has two parts — it works on `lot`, and it does *not* generalize.
The first is now at eight seeds with p = 0.012. The second rests on three seeds
for `iid` and `lot_time`; only `size` has been taken to eight (p = 0.162 on
macro-F1, p = 0.202 on `Scratch`, unestablished either way).

A "does not generalize" claim built on the sample size that has been wrong four
times this weekend is not a claim, it is the absence of one. The whole point of
the finding is the contrast between protocols, so both sides of the contrast
need the same instrument — the same argument as §51, applied to the part of my
own result that happens to be convenient.

`scripts/pooling_protocols_seeds.sh` takes `iid` and `lot_time` to eight seeds
per arm and runs the permutation test on macro-F1 and on `Scratch` for each,
every step behind `verify_stage.py`.

**H54, before the run:** `iid` comes out positive but weaker than `lot` and does
not reach p < 0.05 — its three-seed effect was +0.0090 against `lot`'s +0.0173,
and its baseline is higher. `lot_time` is a genuine null. If `iid` *does* reach
significance the claim simplifies to "helps where geometry is shared, hurts
where it is not", which is stronger and cleaner than what I have now, so this is
one of the few predictions here I would be pleased to lose.

### 56. The Monday document was reporting the weekend's only positive result as unestablished

`WEEKEND.md` §2.0 builds its verdicts from the range test. At three seeds that
read *separated* on both metrics. The `lot` sweep has since gone to eight, and
the same code now reads:

| pooling | macro-F1 | vs `mean` | verdict | Scratch | verdict |
|---|---|---|---|---|---|
| `meanmax` | 0.8888 (n=8) | +0.0150 | **overlaps** | 0.7764 | **overlaps** |

So the hand-off was reporting an effect with a permutation p of **0.00047** as
not established, because the criterion it uses becomes stricter as data
accumulates. Nothing was stale and no number was wrong — the document
regenerated faithfully from current cells, and the conclusion it drew from them
inverted while the evidence got stronger.

That is a nastier version of §53. There the hazard was a partial sample making a
number move; here it is a *correct* number, from a *complete* sample, run
through a criterion whose failure mode is invisible at the point of use. A
reader of §2.0 would have seen "overlaps" twice and concluded the weekend
produced nothing.

Fixed: §2.0 now prints the permutation table under an instruction to ignore the
range verdicts above it, with the reason. The range criterion stays in the
tables it is right for — dozens of three-seed cells where a p-value is
unavailable at any effect size — and is labelled a floor rather than a verdict.

### 57. The control for that result is still at three seeds

The claim is not "mean-and-max scores higher"; it is "mean-and-max scores higher
*and it is not the extra parameters*". The second half rests entirely on
`meanmean` — the mean concatenated with itself, identical parameter count, no
extra information — which is at **three** seeds while the treatment is at eight.

So the load-bearing control for the weekend's only positive result sits at the
sample size that has been wrong four times, supporting a result measured at
nearly three times that. This is the third appearance of the same asymmetry and
the first in the genuinely convenient direction: a control that buys nothing at
n = 3 is exactly what the claim wants, and I did not extend it while extending
everything around it.

`scripts/pooling_control_seeds.sh` takes it to eight and runs the permutation
test on both metrics.

**H56, before the run:** the control stays null. Its three-seed deltas were
−0.0020 macro-F1 and −0.0082 `Scratch`, both slightly negative, and there is no
mechanism by which duplicating a vector adds information — the extra head
parameters see a copy of what they already had. If it comes out positive and
significant, the pooling result is a capacity result, the mechanism story is
wrong, and the weekend's one positive finding becomes "a wider classification
head helps", which is a much less interesting sentence and would have to be the
one printed.

### 58. The most-read paragraph described a methodology two later sections contradict

Reading `WEEKEND.md` end to end deliberately — rather than catching things
incidentally, which is how the last three of these were found — the opening
"Read this first" was wrong in two ways, and it is the paragraph most likely to
be the only one anyone reads.

**It stated the floor as one number.** *"The same cell, run 6 times with
identical arguments, spans 0.0054 macro-F1. That is the measurement floor of
this pipeline."* §28 established that it is not a pipeline property but a
protocol property, measured `lot` 0.0054 and `size` 0.0133, and built
`floors()`/`floor_for()` to use each protocol's own. The opening had never been
updated, so the document introduced a global constant and then, forty lines
later, judged `size` cells against a different one without saying why.

**It described the range test as the verdict.** *"Every verdict below requires
two cells' seed ranges to be disjoint and the margin between them to clear that
floor."* §56 established that this criterion becomes stricter as evidence
accumulates and calls an effect with p = 0.00047 *overlapping*, and §2.0 of the
same document now instructs the reader to ignore it there. A document that opens
by endorsing a criterion and later tells you to disregard it has not made a
correction; it has made a contradiction.

Rewritten to state both floors, that they differ by a factor of two and a half,
and the two-tier standard explicitly: the range test is a **screen** for tables
of dozens of three-seed cells where a p-value is unavailable at any effect size,
and the permutation test is the **verdict** wherever a comparison has been run
at eight seeds per arm. Also fixed `weekend.py` importing `floors` and never
calling it, which is why the opening had a hardcoded single value at all.

**On the method that found it.** The previous three staleness bugs — the "no
positive results" line, the `size`-half-stands claim, the range verdict on the
pooling table — were all caught while doing something else. This one was found
by reading the deliverable straight through against current data, which took
about four minutes and is the obvious thing to do before handing something over.
The generators guarantee that no number is typed by hand. They guarantee nothing
about whether the sentences around the numbers still describe them, and after
fifty-eight entries the ratio is not reassuring: every prose claim in these
documents that I have checked against the data it sits beside has needed
revision at least once.
