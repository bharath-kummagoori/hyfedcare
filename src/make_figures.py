"""
Figure generation for the HyFedCare chapter.

Reads results/experiment_results.json (produced by hyfedcare_experiment.py)
and regenerates all five figures used in the chapter, each as a PNG for
quick viewing and a 300-dpi TIFF for the publisher.

Run from the repository root, after the experiment script:

    python3 src/make_figures.py

Figures land in results/figures/. Diagrams (Figures 1-3) are drawn
programmatically so the whole paper trail stays in code; charts (Figures
4-5) are built from the results file, never from hand-typed numbers.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Okabe-Ito palette: colorblind-safe, print-safe.
BLUE, ORANGE, GREEN = '#0072B2', '#E69F00', '#009E73'
VERM, PURPLE, GREY = '#D55E00', '#CC79A7', '#666666'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})

RESULTS = os.path.join('results', 'experiment_results.json')
OUTDIR = os.path.join('results', 'figures')


def save(fig, name):
    """Write one figure as PNG (screen) and LZW-compressed TIFF (print)."""
    fig.savefig(os.path.join(OUTDIR, name + '.png'), dpi=200,
                bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(OUTDIR, name + '.tif'), dpi=300,
                bbox_inches='tight', facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def box(ax, x, y, w, h, label, fc, fontsize=8.6, ec='#333333'):
    """Rounded labelled box - the basic building block of the diagrams."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, fc=fc, ec=ec, lw=1.0,
                 boxstyle='round,pad=0.06,rounding_size=0.10'))
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
            fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2, color='#333333', lw=1.3, ls='-'):
    """Directed arrow between two points."""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=13, color=color, lw=lw, linestyle=ls))


def fig1_architecture():
    """Figure 1: the three-tier device / edge / coordination architecture."""
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    # Background bands, one per tier.
    for (y0, h, c) in [(0.3, 2.6, '#EAF3FA'), (3.3, 2.8, '#FDF3E3'),
                       (6.5, 3.2, '#E9F7F1')]:
        ax.add_patch(FancyBboxPatch((0.15, y0), 9.7, h, fc=c, ec='none',
                     boxstyle='round,pad=0.04'))
    ax.text(0.35, 2.62, 'TIER 1 - DEVICE (IoMT)', fontsize=8.5,
            fontweight='bold', color=BLUE)
    ax.text(0.35, 5.82, 'TIER 2 - EDGE (hospital gateways)', fontsize=8.5,
            fontweight='bold', color='#B07600')
    ax.text(0.35, 9.42, 'TIER 3 - COORDINATION (cloud aggregation service)',
            fontsize=8.5, fontweight='bold', color=GREEN)

    devices = ['Wearable\nECG patch', 'Insulin\npump', 'Bedside\nmonitor',
               'Imaging\nworkstation', 'Lab info.\nsystem', 'Pharmacy\nrecords']
    for i, lab in enumerate(devices):
        box(ax, 0.5 + i * 1.58, 0.55, 1.35, 1.15, lab, 'white', 7.6)
    ax.text(5.0, 0.18, 'Raw data never leaves the device / institution. '
            'Local DP noise + secure-aggregation masking; split learning '
            'for constrained devices.', fontsize=7.4, ha='center',
            style='italic', color=GREY)

    gateways = ['Hospital A gateway\npartial aggregation\n+ distributed DP',
                'Hospital B gateway\npartial aggregation\n+ distributed DP',
                'Consortium gateway\nPPRL entity alignment\n(vertical FL)']
    for x, lab in zip([1.0, 4.2, 7.4], gateways):
        box(ax, x, 3.7, 2.3, 1.7, lab, 'white', 7.8)

    box(ax, 2.6, 7.0, 4.8, 1.9,
        'Global aggregator\nsecure aggregation (masked sums)\n'
        '+ CKKS homomorphic aggregation\n+ robust aggregation (trimmed mean)',
        'white', 8.2)
    box(ax, 7.9, 7.0, 1.85, 1.9, 'Model\nregistry &\naudit log', 'white', 7.8)

    # Devices feed their own institution's gateway (LAN only)...
    dev_x = [1.2, 2.75, 4.35, 5.9, 7.5, 9.05]
    gw_x = [2.0, 2.0, 5.2, 5.2, 8.4, 8.4]
    for x, tx in zip(dev_x, gw_x):
        arrow(ax, x, 1.85, tx, 3.62, color=BLUE, lw=1.0)
    # ...and gateways feed the coordination tier (the only WAN traffic).
    for x in [2.15, 5.35, 8.55]:
        arrow(ax, x, 5.5, 5.0, 6.95, color='#B07600', lw=1.2)
    arrow(ax, 5.0, 6.95, 2.15, 5.55, color=GREY, lw=0.9, ls='--')
    arrow(ax, 7.4, 8.0, 7.9, 8.0, color=GREY, lw=0.9)
    ax.text(9.15, 6.35, 'WAN traffic:\nonly E gateway\nmessages/round',
            fontsize=7.2, ha='center', color=GREY)
    save(fig, 'fig1_architecture')


def fig2_partitioning():
    """Figure 2: horizontal FL, vertical FL and split learning side by side."""
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')

    ax.text(2.0, 5.6, 'Horizontal FL', fontsize=10, fontweight='bold',
            ha='center', color=BLUE)
    for i in range(3):
        box(ax, 0.4, 4.0 - i * 1.35, 3.2, 1.0,
            f'Hospital {chr(65 + i)}: same features,\ndifferent patients',
            '#EAF3FA', 7.6)

    ax.text(6.0, 5.6, 'Vertical FL', fontsize=10, fontweight='bold',
            ha='center', color='#B07600')
    box(ax, 4.45, 3.3, 1.55, 1.7, 'Hospital:\nvitals,\ndiagnoses', '#FDF3E3', 7.4)
    box(ax, 6.10, 3.3, 1.55, 1.7, 'Imaging\ncenter:\nscans', '#FDF3E3', 7.4)
    box(ax, 4.45, 1.35, 3.2, 1.0, 'Pharmacy: prescriptions', '#FDF3E3', 7.6)
    ax.text(6.05, 0.75, 'same patients, complementary features\n'
            '(privacy-preserving record linkage)', fontsize=7.0,
            ha='center', style='italic', color=GREY)

    ax.text(10.0, 5.6, 'Split learning', fontsize=10, fontweight='bold',
            ha='center', color=GREEN)
    box(ax, 8.6, 3.6, 2.8, 1.2,
        'Wearable: shallow layers only\n(raw signal stays on device)',
        '#E9F7F1', 7.4)
    box(ax, 8.6, 1.6, 2.8, 1.2, 'Gateway: deep layers +\nbackpropagation',
        '#E9F7F1', 7.4)
    arrow(ax, 10.0, 3.55, 10.0, 2.9, color=GREEN)
    ax.text(10.0, 3.18, 'smashed activations', fontsize=6.8, ha='center',
            color=GREY)
    save(fig, 'fig2_partitioning')


def fig3_composition_map():
    """Figure 3: the mechanism-to-tier composition map as a table graphic."""
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.axis('off')
    cols = ['Tier', 'Dominant threats', 'Mechanisms applied',
            'Guarantee character', 'Device cost']
    rows = [
        ['Device\n(IoMT)', 'Physical capture;\neavesdropping;\ninference from updates',
         'Local DP noise;\nsecure-aggregation\nmasking; split learning',
         'Local DP (per-client);\ninformation-theoretic\nmasking', 'Milliwatt-\nfeasible'],
        ['Edge\n(gateway)', 'Curious gateway operator;\nlateral movement;\npoisoned clients',
         'Distributed DP (calibrated);\npartial aggregation;\nanomaly scoring',
         'zCDP composition\nacross rounds\n(bounded, quantified)', 'Server-class,\nmodest'],
        ['Coordination\n(cloud)', 'Honest-but-curious\naggregator; collusion;\nmodel-lineage tampering',
         'CKKS additive HE;\ndropout-tolerant secure\naggregation; trimmed mean',
         'Cryptographic\n(computational);\nrobustness (heuristic)', 'High but\ncentralized'],
    ]
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center',
                   cellLoc='left', colWidths=[0.10, 0.24, 0.26, 0.24, 0.12])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1, 3.1)
    band = {1: '#EAF3FA', 2: '#FDF3E3', 3: '#E9F7F1'}
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#CCCCCC')
        if r == 0:
            cell.set_facecolor('#1F3864')
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_height(0.09)
        else:
            cell.set_facecolor(band[r])
    ax.set_title('HyFedCare composition map: mechanism-to-tier assignment',
                 fontsize=11, fontweight='bold', pad=12)
    save(fig, 'fig3_composition_map')


def fig4_tradeoff(res):
    """Figure 4: privacy-utility trade-off, record-level vs client-level DP."""
    def finite(rows):
        return [r for r in rows if np.isfinite(r['epsilon'])]

    cl, rl = finite(res['client_level_central']), finite(res['record_level'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.9))

    # Panel (a): accuracy against epsilon for both granularities.
    ax1.errorbar([r['epsilon'] for r in rl], [r['acc_mean'] for r in rl],
                 yerr=[r['acc_std'] for r in rl], marker='o', color=BLUE,
                 capsize=3, lw=1.7, label='Record-level DP (per patient)')
    ax1.errorbar([r['epsilon'] for r in cl], [r['acc_mean'] for r in cl],
                 yerr=[r['acc_std'] for r in cl], marker='s', color=ORANGE,
                 capsize=3, lw=1.7, label='Client-level DP (per institution)')
    ax1.axhline(res['centralized']['acc'], color=GREY, ls='--', lw=1.1,
                label=f"Centralized ({res['centralized']['acc']:.3f})")
    ax1.axhline(res['local_only']['acc_mean'], color=VERM, ls='-.', lw=1.1,
                label=f"Local-only avg ({res['local_only']['acc_mean']:.3f})")
    ax1.set_xscale('log')
    ax1.set_xlabel('Privacy budget ε  (δ = 10⁻⁵, zCDP)')
    ax1.set_ylabel('Test accuracy')
    ax1.set_title('(a) Utility vs. budget, by DP granularity', fontsize=10)
    ax1.legend(fontsize=7.0, frameon=False, loc='lower right')
    ax1.set_ylim(0.88, 1.0)

    # Panel (b): the MIA floor. Flat lines here are the honest story - a
    # linear model on tabular data barely memorizes, so the weakest attack
    # has nothing to find at any budget.
    ax2.errorbar([r['epsilon'] for r in rl], [r['mia_mean'] for r in rl],
                 yerr=[r['mia_std'] for r in rl], marker='o', color=BLUE,
                 capsize=3, lw=1.6, label='Record-level DP')
    ax2.errorbar([r['epsilon'] for r in cl], [r['mia_mean'] for r in cl],
                 yerr=[r['mia_std'] for r in cl], marker='s', color=ORANGE,
                 capsize=3, lw=1.6, label='Client-level DP')
    ax2.axhline(0.5, color=GREY, ls='--', lw=1.1, label='Random guess (0.500)')
    ax2.set_xscale('log')
    ax2.set_xlabel('Privacy budget ε  (δ = 10⁻⁵, zCDP)')
    ax2.set_ylabel('Membership-inference AUC')
    ax2.set_title('(b) Loss-threshold MIA floor', fontsize=10)
    ax2.legend(fontsize=7.0, frameon=False)
    ax2.set_ylim(0.48, 0.56)
    fig.tight_layout()
    save(fig, 'fig4_tradeoff')


def fig5_communication(res):
    """Figure 5: WAN traffic and message counts, flat vs hierarchical."""
    comm = res['communication']
    rounds = np.arange(0, comm['rounds'] + 1)
    flat_kb = rounds * comm['flat_wan_msgs_per_round'] * comm['model_bytes'] / 1024
    hier_kb = rounds * comm['hier_wan_msgs_per_round'] * comm['model_bytes'] / 1024

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax1.plot(rounds, flat_kb, color=VERM, lw=1.8,
             label=f"Flat FedAvg (K = {res['clients']} clients on WAN)")
    ax1.plot(rounds, hier_kb, color=BLUE, lw=1.8,
             label=f"Hierarchical (E = {res['edges']} gateways on WAN)")
    ax1.set_xlabel('Federated round')
    ax1.set_ylabel('Cumulative WAN traffic (KB)')
    ax1.set_title('(a) Cumulative WAN traffic', fontsize=10)
    ax1.legend(fontsize=7.6, frameon=False)

    bars = ['Flat\n(WAN)', 'Hierarchical\n(WAN)', 'Hierarchical\n(LAN)']
    vals = [comm['flat_wan_msgs_per_round'], comm['hier_wan_msgs_per_round'],
            comm['hier_lan_msgs_per_round']]
    ax2.bar(bars, vals, color=[VERM, BLUE, '#9DC5E3'], width=0.55)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.6, str(v), ha='center', fontsize=8.6)
    ax2.set_ylabel('Messages per round')
    ax2.set_title('(b) Message counts per round', fontsize=10)
    fig.tight_layout()
    save(fig, 'fig5_communication')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    res = json.load(open(RESULTS))
    fig1_architecture()
    fig2_partitioning()
    fig3_composition_map()
    fig4_tradeoff(res)
    fig5_communication(res)
    print('Figures written to', OUTDIR)


if __name__ == '__main__':
    main()
