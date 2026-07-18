# HyFedCare

Companion code for our book chapter *"A Hybrid Federated Learning Framework
With Tier-Adaptive Privacy-Preserving Techniques: Secure IoT-Enabled Smart
Healthcare"*, submitted to the IGI Global edited volume *Cognitive
Technologies for Secure IoT-Enabled Smart Environments and Cyber-Physical
Systems*.

The chapter argues one main point: in cross-silo healthcare FL, privacy
mechanisms should be matched to the tier of the architecture (device, edge,
coordination) instead of being applied uniformly - and the granularity of the
DP guarantee (client-level vs record-level) should be stated honestly,
because the two are not interchangeable. The experiments here back that up
with numbers. The result we found most interesting: record-level DP, the
stronger per-patient guarantee, actually costs *less* utility than
client-level DP at the same budget in the cross-silo regime, simply because
per-example sensitivity is amortized over hundreds of records instead of a
handful of clients.

## What's here

```
src/
  hyfedcare_experiment.py   the three experiment arms + baselines + accounting
  make_figures.py           regenerates every figure in the chapter
data/
  README.md                 where the data comes from (no files needed - see note)
results/
  experiment_results.json   output of the experiment script (committed for reference)
  figures/                  created by make_figures.py, not committed
```

## Running it

You need Python 3 with numpy, scikit-learn and matplotlib. Nothing else.

```
python3 src/hyfedcare_experiment.py
python3 src/make_figures.py
```

The first script takes well under a minute on a laptop and writes
`results/experiment_results.json`. The second reads that file and produces
the five chapter figures (PNG for viewing, 300-dpi TIFF for the publisher).
Everything is seeded, so you should get exactly the numbers reported in the
chapter - if you don't, please open an issue, we'd genuinely like to know.

## The three experiment arms

**Arm 1 - client-level DP-FedAvg.** The usual construction: each client's
whole update is clipped and the sum is noised. The (ε, δ) this produces
protects an entire client's dataset. We label it that way everywhere,
because calling it "per-patient" would be wrong.

**Arm 2 - record-level federated DP-SGD.** Per-example gradients are clipped
before aggregation, so adding or removing one patient record changes the
released sum by at most C_rec. The same accountant now gives a genuine
per-record guarantee. At ε ≈ 4.1 per record this kept 96.6% test accuracy
against a 97.9% centralized oracle.

**Arm 3 - noise placement.** Same noise multiplier, moved from a single
coordination-tier addition to independent per-device additions. Cost: about
ten accuracy points. This is why the framework defaults to edge/coordination
noise placement.

## Honest limitations

This is an illustrative simulation, not a clinical validation: WDBC is small
and tabular, the model is logistic regression, and the membership-inference
attack is the weakest credible one (it serves as a floor measurement - at
this model capacity it finds almost nothing even without DP, and we say so
in the chapter rather than pretending otherwise). The natural next step is
re-running the same protocol on FLamby-style cross-silo benchmarks with
nonlinear models, where the attack suite becomes discriminating.

## Authors

- Kummagoori Bharath - School of Computer Science and Engineering, Lovely
  Professional University, Phagwara, Punjab, India
- Dr. Pooja Chopra - School of Computer Application, Lovely Professional
  University, Phagwara, Punjab, India
- Shikha Khullar

Questions or issues: bharath.95k@gmail.com, or open an issue here.
