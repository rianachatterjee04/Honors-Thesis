import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from io import StringIO
import re

# ── Load data ──────────────────────────────────────────────────────────────────
def load(path):
    with open(path, 'rb') as f:
        raw = f.read().replace(b'\x00', b'')
    text = raw.decode('utf-8', errors='replace')
    text = re.sub(r'(\d)\r?\n(\d)', r'\1\n\2', text)
    df = pd.read_csv(StringIO(text))
    df.columns = df.columns.str.strip()
    for col in ['threshold', 'latency', 'usage', 'user_score', 'reward']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['threshold', 'user_score', 'latency', 'usage'])
    return df

DATA = {
    'AMD':      {'live': load('/home/claude/data/AMD/AMD_analysis/bo_history.csv'),
                 'offline': load('/home/claude/data/AMD/AMD_analysis/bo_history_offline.csv')},
    'DR':       {'live': load('/home/claude/data/DR/DR_analysis/bo_history.csv'),
                 'offline': load('/home/claude/data/DR/DR_analysis/bo_history_offline.csv')},
    'RP':       {'live': load('/home/claude/data/RP/RP_analysis/bo_history.csv'),
                 'offline': load('/home/claude/data/RP/RP_analysis/bo_history_offline.csv')},
    'Glaucoma': {'live': load('/home/claude/data/glaucoma/glaucoma_analysis/bo_history.csv'),
                 'offline': load('/home/claude/data/glaucoma/glaucoma_analysis/bo_history_offline.csv')},
}

FULL = {
    'AMD':      'Age-Related Macular Degeneration',
    'DR':       'Diabetic Retinopathy',
    'RP':       'Retinitis Pigmentosa',
    'Glaucoma': 'Glaucoma',
}

# Reward formula (from bayesian_optimization.py)
#   latency_penalty = -min(latency/1000, 1.0)
#   usage_penalty   = -(usage * 5.0)
#   human_reward    = user_score / 2.0
#   reward          = latency_penalty + usage_penalty + human_reward

def compute_components(df):
    df = df.copy()
    df['latency_penalty'] = -df['latency'].clip(upper=1000.0) / 1000.0
    df['usage_penalty']   = -(df['usage'] * 5.0)
    df['human_reward']    = df['user_score'] / 2.0
    return df

# ── Smooth curve: interpolate on fine grid then Gaussian smooth ───────────────
def smooth_curve(thresh, values, n=400, sigma=12):
    order = np.argsort(thresh)
    t_s = thresh[order]
    v_s = values[order]
    t_new = np.linspace(0.0, 1.0, n)
    v_interp = np.interp(t_new, t_s, v_s,
                         left=v_s[0], right=v_s[-1])
    v_smooth = gaussian_filter1d(v_interp, sigma=sigma)
    return t_new, v_smooth

# ── One figure per condition ────────────────────────────────────────────────────
def make_chart(cond):
    live_raw    = compute_components(DATA[cond]['live'])
    offline_raw = compute_components(DATA[cond]['offline'])

    # Combine both phases; average duplicates at same threshold
    combined = pd.concat([live_raw, offline_raw], ignore_index=True)
    combined = combined.groupby('threshold', as_index=False)[
        ['latency_penalty', 'usage_penalty', 'human_reward']
    ].mean()

    t = combined['threshold'].values
    lat = combined['latency_penalty'].values
    usg = combined['usage_penalty'].values
    hum = combined['human_reward'].values

    t_smooth, lat_smooth = smooth_curve(t, lat)
    _,        usg_smooth = smooth_curve(t, usg)
    _,        hum_smooth = smooth_curve(t, hum)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(t_smooth, lat_smooth, color='#1f77b4', linewidth=2.2,
            label='Latency Penalty')
    ax.plot(t_smooth, usg_smooth, color='#ff7f0e', linewidth=2.2,
            label='Usage Penalty')
    ax.plot(t_smooth, hum_smooth, color='#2ca02c', linewidth=2.2,
            label='Human Reward')

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel('YOLO Confidence Threshold (x)', fontsize=11)
    ax.set_ylabel('Reward Component Value', fontsize=11)
    ax.set_title(f'{cond}  ({FULL[cond]})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9.5, loc='upper left', framealpha=0.9)
    ax.grid(True, linestyle='-', linewidth=0.6, color='#cccccc')
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = f'/home/claude/pareto_simple_{cond.lower()}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')

for cond in ['AMD', 'DR', 'RP', 'Glaucoma']:
    make_chart(cond)

print('All done.')