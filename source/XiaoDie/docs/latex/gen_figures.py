#!/usr/bin/env python3
"""Generate all SCI-quality figures for the gesture recognition report.

Produces PDF + PNG for each figure with Nature/Science-style color palette,
clean modern flowcharts using FancyBboxPatch, and wide waveform figures.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
import csv

# ---------------------------------------------------------------------------
# Nature / Science style color palette
# ---------------------------------------------------------------------------
COLORS = {
    'blue':        '#4E79A7',
    'orange':      '#F28E2B',
    'green':       '#59A14F',
    'red':         '#E15759',
    'purple':      '#B07AA1',
    'teal':        '#76B7B2',
    'gray':        '#9DA3A6',
    'dark':        '#263238',
    'light_blue':  '#E8F1FA',
    'light_green': '#EAF4EA',
}

# Extended palette for charts (5-class)
BAR_COLORS = [COLORS['blue'], COLORS['orange'], COLORS['green'],
              COLORS['purple'], COLORS['gray']]

# Waveform 3-channel colors (red/green/blue but saturated & pleasant)
WAVE3 = ['#E74C3C', '#27AE60', '#2980B9']

# ---------------------------------------------------------------------------
# Global rcParams -- clean sans-serif, Nature-quality defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          11,
    'axes.labelsize':     12,
    'axes.titlesize':     13,
    'legend.fontsize':    10,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'grid.linewidth':     0.5,
    'lines.linewidth':    0.9,
    'axes.linewidth':     0.8,
    'axes.edgecolor':     '#333333',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '#CCCCCC',
})

DATA_ROOT = Path(__file__).parent.parent / 'VeriHealthi_IMU_Dataset'
OUT = Path(__file__).parent


# ===================================================================
# Data I/O helpers
# ===================================================================
def read_imu(path):
    """Read 6-axis IMU file (5 header lines, then 1 value per line, 7 cols)."""
    raw = np.loadtxt(path, skiprows=5, dtype=np.float32).reshape(-1, 7)[:, :6]
    gyro  = raw[:, :3] / 16.4     # raw -> dps
    accel = raw[:, 3:] / 4096.0   # raw -> g
    return gyro, accel


def read_labels(csv_path):
    """Read label CSV (event_time_s, label)."""
    labels = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append((float(row['event_time_s']), row['label']))
    return labels


def _save(fig, name):
    """Save figure as PDF + PNG and close."""
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  {name} done')


# ===================================================================
# Flowchart drawing helpers  (FancyBboxPatch + FancyArrowPatch)
# ===================================================================
def _add_box(ax, x, y, w, h, text, *,
             fc='#FFFFFF', ec=COLORS['blue'], lw=1.2,
             fontsize=9, fontcolor='#2C3E50', fontweight='bold',
             boxstyle='round,pad=0.3'):
    """Add a rounded box with centered text."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=boxstyle,
                         facecolor=fc, edgecolor=ec, linewidth=lw,
                         zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight,
            color=fontcolor, zorder=3)
    return box


def _add_arrow(ax, x1, y1, x2, y2, *,
               color='#555555', lw=1.2, style='-|>',
               connectionstyle='arc3,rad=0'):
    """Add a straight or curved arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=12,
        color=color,
        linewidth=lw,
        connectionstyle=connectionstyle,
        zorder=1,
    )
    ax.add_patch(arrow)
    return arrow


def _add_label(ax, x, y, text, *, fontsize=7.5, color='#7B68EE',
               ha='center', va='bottom', fontstyle='italic'):
    """Add a small annotation label near an arrow."""
    ax.text(x, y, text, ha=ha, va=va,
            fontsize=fontsize, color=color, fontstyle=fontstyle, zorder=4)


# ===================================================================
#  Fig 1: Raw IMU waveform -- PINCH  (wide, two-column)
# ===================================================================
def fig_raw_imu():
    path     = DATA_ROOT / 'pinch' / 'IMU_pinch_left_2026_05_26_14_18_21_ID0.txt'
    csv_path = DATA_ROOT / 'pinch' / 'pinch_left_2026_05_26_14_18_21_ID0.csv'
    gyro, accel = read_imu(path)
    labels = read_labels(csv_path)
    t = np.arange(len(gyro)) / 50.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    for i, lbl in enumerate(['$g_x$', '$g_y$', '$g_z$']):
        axes[0].plot(t, gyro[:, i], color=WAVE3[i], linewidth=0.5, alpha=0.85, label=lbl)
    for lt, _ in labels:
        if lt < t[-1]:
            axes[0].axvline(lt, color=COLORS['orange'], alpha=0.4, linewidth=0.7, linestyle='--')
    axes[0].set_ylabel('Angular velocity (dps)')
    axes[0].legend(loc='upper right', framealpha=0.9, ncol=3)
    axes[0].set_title('(a) Gyroscope', fontweight='bold', loc='left')

    for i, lbl in enumerate(['$a_x$', '$a_y$', '$a_z$']):
        axes[1].plot(t, accel[:, i], color=WAVE3[i], linewidth=0.5, alpha=0.85, label=lbl)
    for lt, _ in labels:
        if lt < t[-1]:
            axes[1].axvline(lt, color=COLORS['orange'], alpha=0.4, linewidth=0.7, linestyle='--')
    axes[1].set_ylabel('Acceleration (g)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(loc='upper right', framealpha=0.9, ncol=3)
    axes[1].set_title('(b) Accelerometer', fontweight='bold', loc='left')

    fig.tight_layout()
    _save(fig, 'fig_raw_imu_pinch')


# ===================================================================
#  Fig NEW: Raw IMU waveform -- CLENCH  (wide)
# ===================================================================
def fig_clench_waveform():
    path     = DATA_ROOT / 'clench' / 'IMU_clench_left_2026_05_26_14_23_39_ID0.txt'
    csv_path = DATA_ROOT / 'clench' / 'clench_left_2026_05_26_14_23_39_ID0.csv'
    gyro, accel = read_imu(path)
    labels = read_labels(csv_path)
    t = np.arange(len(gyro)) / 50.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    for i, lbl in enumerate(['$g_x$', '$g_y$', '$g_z$']):
        axes[0].plot(t, gyro[:, i], color=WAVE3[i], linewidth=0.5, alpha=0.85, label=lbl)
    for lt, _ in labels:
        if lt < t[-1]:
            axes[0].axvline(lt, color=COLORS['orange'], alpha=0.4, linewidth=0.7, linestyle='--')
    axes[0].set_ylabel('Angular velocity (dps)')
    axes[0].legend(loc='upper right', framealpha=0.9, ncol=3)
    axes[0].set_title('(a) Gyroscope -- Clench Recording', fontweight='bold', loc='left')

    for i, lbl in enumerate(['$a_x$', '$a_y$', '$a_z$']):
        axes[1].plot(t, accel[:, i], color=WAVE3[i], linewidth=0.5, alpha=0.85, label=lbl)
    for lt, _ in labels:
        if lt < t[-1]:
            axes[1].axvline(lt, color=COLORS['orange'], alpha=0.4, linewidth=0.7, linestyle='--')
    axes[1].set_ylabel('Acceleration (g)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(loc='upper right', framealpha=0.9, ncol=3)
    axes[1].set_title('(b) Accelerometer -- Clench Recording', fontweight='bold', loc='left')

    fig.tight_layout()
    _save(fig, 'fig_clench_waveform')


# ===================================================================
#  Fig NEW: Raw IMU waveform -- UP/DOWN  (wide)
# ===================================================================
def fig_updown_waveform():
    path = DATA_ROOT / 'up' / 'IMU_updown_left_2026_05_26_14_34_18_ID0.txt'
    csv_path = DATA_ROOT / 'up' / 'updown_left_2026_05_26_14_34_18_ID0.csv'
    gyro, accel = read_imu(path)
    labels = read_labels(csv_path)
    t = np.arange(len(gyro)) / 50.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    for i, lbl in enumerate(['$g_x$', '$g_y$', '$g_z$']):
        axes[0].plot(t, gyro[:, i], color=WAVE3[i], linewidth=0.5, alpha=0.85, label=lbl)
    for lt, _ in labels:
        if lt < t[-1]:
            axes[0].axvline(lt, color=COLORS['teal'], alpha=0.4, linewidth=0.7, linestyle='--')
    axes[0].set_ylabel('Angular velocity (dps)')
    axes[0].legend(loc='upper right', framealpha=0.9, ncol=3)
    axes[0].set_title('(a) Gyroscope -- Up/Down Recording', fontweight='bold', loc='left')

    for i, lbl in enumerate(['$a_x$', '$a_y$', '$a_z$']):
        axes[1].plot(t, accel[:, i], color=WAVE3[i], linewidth=0.5, alpha=0.85, label=lbl)
    for lt, _ in labels:
        if lt < t[-1]:
            axes[1].axvline(lt, color=COLORS['teal'], alpha=0.4, linewidth=0.7, linestyle='--')
    axes[1].set_ylabel('Acceleration (g)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(loc='upper right', framealpha=0.9, ncol=3)
    axes[1].set_title('(b) Accelerometer -- Up/Down Recording', fontweight='bold', loc='left')

    fig.tight_layout()
    _save(fig, 'fig_updown_waveform')


# ===================================================================
#  Fig 2: Four-gesture magnitude comparison  (wide)
# ===================================================================
def fig_four_gestures():
    configs = [
        ('pinch',   DATA_ROOT / 'pinch'  / 'IMU_pinch_left_2026_05_26_14_18_21_ID0.txt',   '(a) Pinch'),
        ('clench',  DATA_ROOT / 'clench' / 'IMU_clench_left_2026_05_26_14_23_39_ID0.txt',   '(b) Clench'),
        ('up/down', DATA_ROOT / 'up'     / 'IMU_updown_left_2026_05_26_14_34_18_ID0.txt',   '(c) Wrist-up / Wrist-down'),
        ('others',  DATA_ROOT / 'others' / sorted((DATA_ROOT / 'others').glob('*.txt'))[0].name, '(d) Others'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9.2))
    axes = axes.flatten()

    for idx, (name, path, title) in enumerate(configs):
        gyro, accel = read_imu(path)
        mag_a = np.linalg.norm(accel, axis=1)
        mag_g = np.linalg.norm(gyro, axis=1)
        t = np.arange(len(gyro)) / 50.0
        end = min(len(t), 750)

        ax = axes[idx]
        ln1 = ax.plot(t[:end], mag_a[:end], color=COLORS['blue'], linewidth=0.7, label='|accel|')
        ax.set_ylabel('|Accel| (g)', color=COLORS['blue'])
        ax.tick_params(axis='y', labelcolor=COLORS['blue'])

        ax2 = ax.twinx()
        ln2 = ax2.plot(t[:end], mag_g[:end], color=COLORS['red'], linewidth=0.45, alpha=0.7, label='|gyro|')
        ax2.set_ylabel('|Gyro| (dps)', color=COLORS['red'])
        ax2.tick_params(axis='y', labelcolor=COLORS['red'])
        ax2.spines['right'].set_visible(True)
        ax2.spines['top'].set_visible(False)

        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper right', fontsize=9, framealpha=0.9)
        ax.set_title(title, fontweight='bold', loc='left')
        ax.set_xlabel('Time (s)')

    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    _save(fig, 'fig_four_gestures')


# ===================================================================
#  Fig 3: Filter bank illustration (improved colors)
# ===================================================================
def fig_filter_bank():
    path = DATA_ROOT / 'pinch' / 'IMU_pinch_left_2026_05_26_14_18_21_ID0.txt'
    gyro_raw, accel_raw = read_imu(path)

    LP8, HP8, LP20 = 0.5010, 0.4980, 0.7154
    data = np.concatenate([gyro_raw, accel_raw], axis=1)
    low  = np.zeros(6, dtype=np.float32)
    high = np.zeros(6, dtype=np.float32)
    mid  = np.zeros(6, dtype=np.float32)
    prev = np.zeros(6, dtype=np.float32)
    init = False
    mid_energy, low_energy, raw_energy = [], [], []
    for sample in data:
        if not init:
            low = sample.copy(); prev = sample.copy(); init = True
        else:
            low  += LP8  * (sample - low)
            high  = HP8  * (high + sample - prev)
            prev  = sample.copy()
            mid  += LP20 * (high - mid)
        mid_energy.append(np.linalg.norm(mid[3:]))
        low_energy.append(abs(np.linalg.norm(low[3:]) - 1.0))
        raw_energy.append(np.linalg.norm(sample[3:]))

    t = np.arange(len(mid_energy)) / 50.0
    end = 750

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t[:end], raw_energy[:end], color=COLORS['gray'], linewidth=0.5,
            alpha=0.55, label='Raw |accel|')
    ax.plot(t[:end], mid_energy[:end], color=COLORS['red'], linewidth=1.1,
            label='Mid-band (8–20 Hz)')
    ax.plot(t[:end], low_energy[:end], color=COLORS['blue'], linewidth=1.1,
            label='Low-pass detrend')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy')
    ax.set_title('Filter Bank Decomposition on Pinch Recording', fontweight='bold', loc='left')
    ax.legend(loc='upper right', framealpha=0.9)
    fig.tight_layout()
    _save(fig, 'fig_filter_bank')


# ===================================================================
#  Fig 4: Neural network architecture  (clean flowchart)
# ===================================================================
def fig_nn_architecture():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(-0.3, 13.3)
    ax.set_ylim(-0.5, 6.0)
    ax.axis('off')
    ax.set_title('Multi-Model Inference Pipeline', fontweight='bold',
                 fontsize=14, loc='left', pad=12)

    # Box definitions: (x, y, w, h, text, border_color, fill_color)
    BW, BH = 2.0, 1.2   # default box size
    boxes = [
        # Col 0 -- input
        (0.0,  2.4, 1.8, BH, 'IMU\nSamples',
         COLORS['dark'], '#F0F4F8'),
        # Col 1 -- feature extraction
        (2.5,  2.4, 2.0, BH, 'Feature\nExtraction\n(85-dim)',
         COLORS['blue'], COLORS['light_blue']),
        # Col 2 top -- global NN
        (5.2,  3.8, 2.2, BH, 'Global NN\n85→64→32→5',
         COLORS['orange'], '#FFF3E0'),
        # Col 2 bot -- PC head
        (5.2,  1.0, 2.2, BH, 'PC Head\n85→32→16→2',
         COLORS['teal'], COLORS['light_green']),
        # Col 3 top -- TCN guard
        (8.1,  3.8, 2.0, BH, 'TCN Guard\n(8\xd77→5)',
         COLORS['red'], '#FFEAEA'),
        # Col 3 bot -- Template guard
        (8.1,  1.0, 2.0, BH, 'Template\nGuard',
         COLORS['purple'], '#F0E6FF'),
        # Col 4 -- decision
        (10.8, 2.4, 1.8, BH, 'Decision\nLogic',
         '#C0392B', '#FDE8E8'),
    ]

    for x, y, w, h, txt, ec, fc in boxes:
        _add_box(ax, x, y, w, h, txt, fc=fc, ec=ec, lw=1.4, fontsize=9)

    # Arrows  (from right edge -> left edge of next box)
    arrows = [
        # input -> feature
        (1.8,  3.0,  2.5,  3.0),
        # feature -> global NN (up)
        (4.5,  3.2,  5.2,  4.4),
        # feature -> PC head (down)
        (4.5,  2.8,  5.2,  1.6),
        # global NN -> TCN guard
        (7.4,  4.4,  8.1,  4.4),
        # PC head -> template guard
        (7.4,  1.6,  8.1,  1.6),
        # TCN guard -> decision (down-right)
        (10.1, 4.2,  10.8, 3.2),
        # template guard -> decision (up-right)
        (10.1, 1.8,  10.8, 2.8),
        # cross: global NN -> template guard (dashed, will add separately)
    ]
    for x1, y1, x2, y2 in arrows:
        _add_arrow(ax, x1, y1, x2, y2, color=COLORS['dark'], lw=1.3)

    # Cross-connection: global NN output -> template guard (curved)
    _add_arrow(ax, 7.4, 4.0, 8.1, 2.0,
               color=COLORS['gray'], lw=0.9,
               connectionstyle='arc3,rad=-0.3', style='-|>')

    fig.tight_layout()
    _save(fig, 'fig_nn_architecture')


# ===================================================================
#  Fig 5: F1 score bar chart  (QEMU validation)
# ===================================================================
def fig_f1_bar():
    classes = ['pinch', 'clench', 'up', 'down']
    tp = [23, 21, 21, 21]
    fp = [1,  0,  0,  0]
    fn = [1,  0,  0,  0]
    precision = [t_/(t_+f_) if (t_+f_) > 0 else 0 for t_, f_ in zip(tp, fp)]
    recall    = [t_/(t_+n_) if (t_+n_) > 0 else 0 for t_, n_ in zip(tp, fn)]
    f1        = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(precision, recall)]

    x = np.arange(len(classes))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - width, precision, width, label='Precision',
                   color=COLORS['blue'],   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x,         recall,    width, label='Recall',
                   color=COLORS['teal'],   edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, f1,        width, label='F1 Score',
                   color=COLORS['orange'], edgecolor='white', linewidth=0.5)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f'{h:.4f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color=COLORS['dark'])

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel('Score')
    ax.set_ylim(0.90, 1.025)
    ax.set_title('QEMU Validation: Per-Class Precision / Recall / F1',
                 fontweight='bold', loc='left')
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    _save(fig, 'fig_f1_bar')


# ===================================================================
#  Fig 6: Confusion matrix heatmap (QEMU)
# ===================================================================
def fig_confusion_matrix():
    classes = ['pinch', 'clench', 'up', 'down']
    cm = np.array([
        [23, 0, 0, 0],
        [ 0,21, 0, 0],
        [ 0, 0,21, 0],
        [ 0, 0, 0,21],
    ], dtype=np.float32)
    totals = np.array([24, 21, 21, 21], dtype=np.float32)
    cm_norm = cm / totals[:, None]

    fig, ax = plt.subplots(figsize=(6, 5))
    # Use a custom sequential colormap: white -> teal
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'teal_seq', ['#FFFFFF', COLORS['light_blue'], COLORS['blue'], '#1A3A6B'])
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect='equal')

    for i in range(4):
        for j in range(4):
            val = cm_norm[i, j]
            count = int(cm[i, j])
            color = 'white' if val > 0.5 else COLORS['dark']
            ax.text(j, i, f'{count}\n({val:.2f})',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title('QEMU Confusion Matrix', fontweight='bold', loc='left')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Recall')
    cbar.outline.set_linewidth(0.5)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    fig.tight_layout()
    _save(fig, 'fig_confusion_matrix')


# ===================================================================
#  Fig 7: Resource usage  (donut charts, SCI colors)
# ===================================================================
def fig_resource_usage():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Flash: 3 MB
    flash_total = 3145728
    text_size   = 511628
    data_size   = 58028
    flash_free  = flash_total - text_size - data_size

    sizes_f  = [text_size, data_size, flash_free]
    labels_f = [f'.text\n{text_size/1024:.1f} KB',
                f'.data\n{data_size/1024:.1f} KB',
                f'Free\n{flash_free/1024:.1f} KB']
    colors_f = [COLORS['blue'], COLORS['orange'], '#E8ECF0']

    wedges, texts, autotexts = axes[0].pie(
        sizes_f, labels=labels_f, colors=colors_f,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 10}, pctdistance=0.72,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(9)
    # Donut hole
    centre = plt.Circle((0, 0), 0.45, fc='white', ec='none')
    axes[0].add_artist(centre)
    axes[0].set_title('(a) Flash Usage (3 MB)', fontweight='bold', fontsize=12)

    # RAM: 256 KB
    ram_total = 262144
    bss_size  = 50842
    data_ram  = 58028
    ram_free  = ram_total - bss_size - data_ram

    sizes_r  = [data_ram, bss_size, ram_free]
    labels_r = [f'.data\n{data_ram/1024:.1f} KB',
                f'.bss\n{bss_size/1024:.1f} KB',
                f'Free\n{ram_free/1024:.1f} KB']
    colors_r = [COLORS['orange'], COLORS['teal'], '#E8ECF0']

    wedges2, texts2, autotexts2 = axes[1].pie(
        sizes_r, labels=labels_r, colors=colors_r,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 10}, pctdistance=0.72,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    for at in autotexts2:
        at.set_fontweight('bold')
        at.set_fontsize(9)
    centre2 = plt.Circle((0, 0), 0.45, fc='white', ec='none')
    axes[1].add_artist(centre2)
    axes[1].set_title('(b) RAM Usage (256 KB)', fontweight='bold', fontsize=12)

    fig.tight_layout()
    _save(fig, 'fig_resource_usage')


# ===================================================================
#  Fig 8: Decision state machine  (clean flowchart)
# ===================================================================
def fig_state_machine():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.8, 5.0)
    ax.axis('off')
    ax.set_title('Gesture Decision State Machine', fontweight='bold',
                 fontsize=14, loc='left', pad=12)

    # State boxes  (x, y, w, h, text, border_color, fill)
    BW, BH = 2.0, 1.2
    states = [
        (0.3,  1.9, 1.8, BH, 'IDLE',
         COLORS['dark'], '#F0F4F8'),
        (3.2,  3.3, 2.2, BH, 'Small-gesture\naccumulating',
         COLORS['blue'], COLORS['light_blue']),
        (3.2,  0.5, 2.2, BH, 'Turn-gesture\nrun counting',
         COLORS['teal'], COLORS['light_green']),
        (7.0,  3.3, 2.0, BH, 'Emit\npinch / clench',
         COLORS['orange'], '#FFF3E0'),
        (7.0,  0.5, 2.0, BH, 'Emit\nup / down',
         COLORS['green'], '#E8F5E9'),
        (10.5, 1.9, 2.2, BH, 'Refractory\n(1.2 s)',
         COLORS['purple'], '#F0E6FF'),
    ]

    for x, y, w, h, txt, ec, fc in states:
        _add_box(ax, x, y, w, h, txt, fc=fc, ec=ec, lw=1.4, fontsize=9)

    # Arrows
    # IDLE -> small-gesture (up-right)
    _add_arrow(ax, 2.1, 2.8, 3.2, 3.9, color=COLORS['dark'])
    _add_label(ax, 2.3, 3.6, 'small logit\n> none', fontsize=7.5, color=COLORS['blue'])

    # IDLE -> turn-gesture (down-right)
    _add_arrow(ax, 2.1, 2.2, 3.2, 1.1, color=COLORS['dark'])
    _add_label(ax, 2.3, 1.3, 'turn logit\n> none', fontsize=7.5, color=COLORS['teal'])

    # small-gesture -> emit pinch/clench
    _add_arrow(ax, 5.4, 3.9, 7.0, 3.9, color=COLORS['dark'])
    _add_label(ax, 6.0, 4.2, '≥ N\nwindows', fontsize=7.5, color=COLORS['orange'])

    # turn-gesture -> emit up/down
    _add_arrow(ax, 5.4, 1.1, 7.0, 1.1, color=COLORS['dark'])
    _add_label(ax, 6.0, 1.5, '≥ 3\nconsecutive', fontsize=7.5, color=COLORS['green'])

    # emit pinch -> refractory
    _add_arrow(ax, 9.0, 3.6, 10.5, 2.8, color=COLORS['dark'])
    # emit up/down -> refractory
    _add_arrow(ax, 9.0, 1.4, 10.5, 2.2, color=COLORS['dark'])

    # refractory -> IDLE (curved back)
    _add_arrow(ax, 11.6, 1.9, 1.2, 1.9,
               color=COLORS['gray'], lw=0.9,
               connectionstyle='arc3,rad=-0.25')
    _add_label(ax, 6.5, 0.0, 'timeout → reset', fontsize=7.5, color=COLORS['gray'])

    fig.tight_layout()
    _save(fig, 'fig_state_machine')


# ===================================================================
#  Fig 9: Python offline F1 replay  (ALL 1.0000)
# ===================================================================
def fig_python_f1():
    classes  = ['pinch', 'clench', 'up', 'down']
    f1_vals  = [1.0000, 1.0000, 1.0000, 1.0000]
    bar_cols = [COLORS['blue'], COLORS['teal'], COLORS['orange'], COLORS['purple']]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(classes, f1_vals, color=bar_cols,
                  edgecolor='white', width=0.55, linewidth=0.5)
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=COLORS['dark'])
    ax.set_ylim(0.97, 1.015)
    ax.set_ylabel('F1 Score')
    ax.set_title('Python Offline Replay: Per-Class F1 (All-Data, macro = 1.0000)',
                 fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    _save(fig, 'fig_python_f1')


# ===================================================================
#  Fig 10: Dataset distribution  (bar + pie)
# ===================================================================
def fig_dataset_dist():
    categories = ['pinch', 'clench', 'up', 'down', 'others']
    counts     = [154, 154, 150, 150, 57]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) bar chart
    bars = axes[0].bar(categories, counts, color=BAR_COLORS,
                       edgecolor='white', width=0.6, linewidth=0.5)
    for bar, c in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, c + 3, str(c),
                     ha='center', fontsize=11, fontweight='bold', color=COLORS['dark'])
    axes[0].set_ylabel('Number of recordings')
    axes[0].set_title('(a) Recordings per gesture class', fontweight='bold', loc='left')
    axes[0].grid(axis='y', alpha=0.25)

    # (b) pie chart (donut style)
    wedges, texts, autotexts = axes[1].pie(
        counts, labels=categories, colors=BAR_COLORS,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 11},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(9)
    centre = plt.Circle((0, 0), 0.40, fc='white', ec='none')
    axes[1].add_artist(centre)
    axes[1].set_title('(b) Class distribution', fontweight='bold', loc='left')

    fig.tight_layout()
    _save(fig, 'fig_dataset_dist')


# ===================================================================
#  Fig 11: Sliding window illustration (improved colors)
# ===================================================================
def fig_sliding_window():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    np.random.seed(42)
    t = np.arange(200) / 50.0
    sig = np.sin(2 * np.pi * 1.2 * t) * 0.5 + np.random.normal(0, 0.08, len(t))
    ax.plot(t, sig, color=COLORS['dark'], linewidth=0.8, alpha=0.7)

    window_size = 70
    window_colors = [COLORS['blue'], COLORS['teal'], COLORS['orange']]
    for i, start in enumerate([0, 10, 20]):
        e = start + window_size
        if e > len(t):
            break
        ax.axvspan(t[start], t[e - 1], alpha=0.13, color=window_colors[i])
        ax.annotate(f'W{i + 1}', xy=(t[(start + e) // 2], 0.72), fontsize=10,
                    ha='center', fontweight='bold', color=window_colors[i])
        # Add thin border lines at window edges
        ax.axvline(t[start], color=window_colors[i], alpha=0.5, linewidth=0.6, linestyle=':')
        ax.axvline(t[e - 1], color=window_colors[i], alpha=0.5, linewidth=0.6, linestyle=':')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal amplitude')
    ax.set_title('Sliding Window Scheme: window = 70 (1.4 s), stride = 10 (0.2 s)',
                 fontweight='bold', loc='left')
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    _save(fig, 'fig_sliding_window')


# ===================================================================
#  Fig 12: Software workflow / architecture  (clean flowchart)
# ===================================================================
def fig_software_flow():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(-0.3, 15.5)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')
    ax.set_title('Embedded Software Architecture', fontweight='bold',
                 fontsize=14, loc='left', pad=12)

    # Boxes: (x, y, w, h, text, border_color, fill_color)
    BW, BH = 2.0, 1.1
    boxes = [
        (0.3,  3.5, BW, BH, 'IMU\nHardware',
         COLORS['dark'], '#F0F4F8'),
        (0.3,  1.5, BW, BH, 'ISR\n(data ready)',
         COLORS['red'], '#FFEAEA'),
        (3.3,  3.5, BW, BH, 'imu_task\n(HAL read)',
         COLORS['blue'], COLORS['light_blue']),
        (3.3,  1.5, BW, BH, 'CRC-8/SMBus\ncompute',
         COLORS['orange'], '#FFF3E0'),
        (6.5,  2.5, 1.8, BH, 'Sample\nFIFO',
         COLORS['teal'], COLORS['light_green']),
        (9.2,  2.5, 2.2, BH, 'algo_task\n(recognizer)',
         COLORS['purple'], '#F0E6FF'),
        (12.3, 2.5, 1.8, BH, 'UART\nprintf',
         COLORS['green'], '#E8F5E9'),
    ]

    for x, y, w, h, txt, ec, fc in boxes:
        _add_box(ax, x, y, w, h, txt, fc=fc, ec=ec, lw=1.4, fontsize=9)

    # Arrows with labels
    # IMU HW -> ISR (down)
    _add_arrow(ax, 1.3, 3.5, 1.3, 2.6, color=COLORS['dark'])
    _add_label(ax, 1.6, 3.0, 'interrupt', fontsize=7.5, color=COLORS['red'])

    # ISR -> imu_task (right)
    _add_arrow(ax, 2.3, 2.0, 3.3, 2.0, color=COLORS['dark'])
    _add_label(ax, 2.8, 2.2, 'EVENT\nSEN_IRQ', fontsize=7, color=COLORS['purple'])

    # IMU HW -> imu_task (right)
    _add_arrow(ax, 2.3, 4.0, 3.3, 4.0, color=COLORS['dark'])
    _add_label(ax, 2.8, 4.2, 'HAL read', fontsize=7.5, color=COLORS['blue'])

    # imu_task -> CRC (down)
    _add_arrow(ax, 4.3, 3.5, 4.3, 2.6, color=COLORS['dark'])

    # CRC -> imu_task (up, verification)
    _add_arrow(ax, 3.9, 2.6, 3.9, 3.5, color=COLORS['gray'], lw=0.8)

    # imu_task -> FIFO (right, angled)
    _add_arrow(ax, 5.3, 3.5, 6.5, 3.1, color=COLORS['dark'])
    _add_label(ax, 5.8, 3.6, 'push', fontsize=7.5, color=COLORS['teal'])

    # FIFO -> algo_task (right)
    _add_arrow(ax, 8.3, 3.0, 9.2, 3.0, color=COLORS['dark'])
    _add_label(ax, 8.6, 3.2, 'pop', fontsize=7.5, color=COLORS['purple'])

    # algo_task -> UART (right)
    _add_arrow(ax, 11.4, 3.0, 12.3, 3.0, color=COLORS['dark'])
    _add_label(ax, 11.7, 3.2, 'result', fontsize=7.5, color=COLORS['green'])

    fig.tight_layout()
    _save(fig, 'fig_software_flow')


# ===================================================================
#  Overrides: compact report-ready flowcharts
# ===================================================================
def fig_nn_architecture():
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.set_xlim(0.0, 11.5)
    ax.set_ylim(0.55, 4.50)
    ax.axis('off')

    boxes = [
        (0.35, 2.05, 1.55, 1.00, 'IMU\nsamples',
         COLORS['dark'], '#F5F7FA'),
        (2.25, 2.05, 1.85, 1.00, 'Feature\nvector\n85 dim',
         COLORS['blue'], COLORS['light_blue']),
        (4.65, 3.25, 1.95, 1.00, 'Global\nclassifier\n85-64-32-5',
         COLORS['orange'], '#FFF3E6'),
        (4.65, 0.90, 1.95, 1.00, 'Pinch/clench\nclassifier\n85-32-16-2',
         COLORS['teal'], '#E7F3F2'),
        (7.05, 3.25, 1.95, 1.00, 'Temporal\ncheck\n8 windows',
         COLORS['red'], '#FCEBEB'),
        (7.05, 0.90, 1.95, 1.00, 'Centroid\ncheck\n16 features',
         COLORS['purple'], '#F4EDF4'),
        (9.65, 2.05, 1.55, 1.00, 'Event\nstate\nmachine',
         COLORS['green'], '#ECF5EA'),
    ]

    for x, y, w, h, txt, ec, fc in boxes:
        _add_box(ax, x, y, w, h, txt, fc=fc, ec=ec, lw=1.35,
                 fontsize=8.6, boxstyle='round,pad=0.22')

    arrows = [
        (1.90, 2.55, 2.25, 2.55),
        (4.10, 2.75, 4.65, 3.75),
        (4.10, 2.35, 4.65, 1.40),
        (6.60, 3.75, 7.05, 3.75),
        (6.60, 1.40, 7.05, 1.40),
        (9.00, 3.75, 9.65, 2.85),
        (9.00, 1.40, 9.65, 2.25),
    ]
    for x1, y1, x2, y2 in arrows:
        _add_arrow(ax, x1, y1, x2, y2, color=COLORS['dark'], lw=1.15)

    _add_label(ax, 4.42, 3.22, 'candidate', fontsize=7.0,
               color=COLORS['orange'])
    _add_label(ax, 4.42, 1.72, 'small gesture', fontsize=7.0,
               color=COLORS['teal'])
    _add_label(ax, 9.35, 3.38, 'scores', fontsize=7.0,
               color=COLORS['red'])
    _add_label(ax, 9.35, 1.72, 'distance', fontsize=7.0,
               color=COLORS['purple'])

    fig.tight_layout()
    _save(fig, 'fig_nn_architecture')


def fig_state_machine():
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.set_xlim(0.0, 11.5)
    ax.set_ylim(0.15, 4.35)
    ax.axis('off')

    boxes = [
        (0.35, 1.85, 1.60, 1.00, 'IDLE',
         COLORS['dark'], '#F5F7FA'),
        (2.70, 3.10, 2.00, 1.00, 'Small gesture\naccumulation',
         COLORS['blue'], COLORS['light_blue']),
        (2.70, 0.60, 2.00, 1.00, 'Turn gesture\nrun count',
         COLORS['teal'], '#E7F3F2'),
        (5.85, 3.10, 1.80, 1.00, 'Emit\npinch/clench',
         COLORS['orange'], '#FFF3E6'),
        (5.85, 0.60, 1.80, 1.00, 'Emit\nup/down',
         COLORS['green'], '#ECF5EA'),
        (8.90, 1.85, 1.90, 1.00, 'Refractory\n1200 ms',
         COLORS['purple'], '#F4EDF4'),
    ]
    for x, y, w, h, txt, ec, fc in boxes:
        _add_box(ax, x, y, w, h, txt, fc=fc, ec=ec, lw=1.35,
                 fontsize=8.8, boxstyle='round,pad=0.22')

    arrows = [
        (1.95, 2.65, 2.70, 3.60),
        (1.95, 2.05, 2.70, 1.10),
        (4.70, 3.60, 5.85, 3.60),
        (4.70, 1.10, 5.85, 1.10),
        (7.65, 3.35, 8.90, 2.60),
        (7.65, 1.35, 8.90, 2.10),
    ]
    for x1, y1, x2, y2 in arrows:
        _add_arrow(ax, x1, y1, x2, y2, color=COLORS['dark'], lw=1.15)
    _add_arrow(ax, 9.85, 1.85, 1.10, 1.85, color=COLORS['gray'],
               lw=0.95, connectionstyle='arc3,rad=-0.25')

    _add_label(ax, 2.30, 3.18, 'pinch/clench', fontsize=7.0,
               color=COLORS['blue'])
    _add_label(ax, 2.25, 1.35, 'up/down', fontsize=7.0,
               color=COLORS['teal'])
    _add_label(ax, 5.28, 3.88, 'enough windows', fontsize=7.0,
               color=COLORS['orange'])
    _add_label(ax, 5.28, 1.38, '3 consecutive', fontsize=7.0,
               color=COLORS['green'])
    _add_label(ax, 5.55, 0.22, 'timeout -> IDLE', fontsize=7.0,
               color=COLORS['gray'])

    fig.tight_layout()
    _save(fig, 'fig_state_machine')


def fig_software_flow():
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0.0, 11.5)
    ax.set_ylim(0.65, 4.55)
    ax.axis('off')

    boxes = [
        (0.45, 3.45, 1.70, 0.95, 'IMU\nhardware',
         COLORS['dark'], '#F5F7FA'),
        (0.45, 1.10, 1.70, 0.95, 'ISR\ndata ready',
         COLORS['red'], '#FCEBEB'),
        (3.00, 3.45, 1.85, 0.95, 'imu_task\nHAL read',
         COLORS['blue'], COLORS['light_blue']),
        (3.00, 1.10, 1.85, 0.95, 'CRC-8\nSMBus',
         COLORS['orange'], '#FFF3E6'),
        (5.65, 2.25, 1.65, 0.95, 'Sample\nFIFO',
         COLORS['teal'], '#E7F3F2'),
        (8.00, 2.25, 1.75, 0.95, 'algo_task\nrecognizer',
         COLORS['purple'], '#F4EDF4'),
        (10.20, 2.25, 1.05, 0.95, 'UART\nBLE',
         COLORS['green'], '#ECF5EA'),
    ]

    for x, y, w, h, txt, ec, fc in boxes:
        _add_box(ax, x, y, w, h, txt, fc=fc, ec=ec, lw=1.35,
                 fontsize=8.8, boxstyle='round,pad=0.22')

    arrows = [
        (1.30, 3.45, 1.30, 2.05),
        (2.15, 3.92, 3.00, 3.92),
        (2.15, 1.58, 3.00, 1.58),
        (3.92, 3.45, 3.92, 2.05),
        (4.85, 3.65, 5.65, 2.85),
        (4.85, 1.55, 5.65, 2.45),
        (7.30, 2.72, 8.00, 2.72),
        (9.75, 2.72, 10.20, 2.72),
    ]
    for x1, y1, x2, y2 in arrows:
        _add_arrow(ax, x1, y1, x2, y2, color=COLORS['dark'], lw=1.15)

    _add_label(ax, 1.60, 2.68, 'interrupt', fontsize=7.0,
               color=COLORS['red'])
    _add_label(ax, 2.55, 4.12, 'frames', fontsize=7.0,
               color=COLORS['blue'])
    _add_label(ax, 5.30, 3.22, 'push', fontsize=7.0,
               color=COLORS['teal'])
    _add_label(ax, 7.65, 2.92, 'pop', fontsize=7.0,
               color=COLORS['purple'])
    _add_label(ax, 9.98, 2.92, 'event', fontsize=7.0,
               color=COLORS['green'])

    fig.tight_layout()
    _save(fig, 'fig_software_flow')


# ===================================================================
#  Main
# ===================================================================
if __name__ == '__main__':
    print('Generating SCI-quality figures...\n')
    fig_raw_imu()
    fig_clench_waveform()
    fig_updown_waveform()
    fig_four_gestures()
    fig_filter_bank()
    fig_nn_architecture()
    fig_f1_bar()
    fig_confusion_matrix()
    fig_resource_usage()
    fig_state_machine()
    fig_python_f1()
    fig_dataset_dist()
    fig_sliding_window()
    fig_software_flow()
    print('\n=== All 14 figures generated ===')
