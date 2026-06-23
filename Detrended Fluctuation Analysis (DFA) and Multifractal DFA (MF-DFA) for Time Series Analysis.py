# ============================================================
# DFA/MF-DFA FRACTAL ANALYSIS FOR TIME SERIES IN QGIS. V.1.0

# ORIGINAL DFA REFERENCE:
# Peng, C.K., Buldyrev, S.V., Havlin, S., Simons, M., Stanley, H.E., 
# Goldberger, A.L. (1994). Mosaic organization of DNA nucleotides. 
# Physical Review E, 49(2), 1685-1689.
# doi:10.1103/physreve.49.1685. PMID 9961383.
#
# MF-DFA REFERENCE:
# Kantelhardt, J.W., Zschiegner, S.A., Koscielny-Bunde, E., Havlin, S., 
# Bunde, A., Stanley, H.E. (2002). Multifractal detrended fluctuation 
# analysis of nonstationary time series. Physica A, 316(1-4), 87-114.
#
# BOOTSTRAP REFERENCE:
# Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap. 
# Journal of the American Statistical Association, 89(428), 1303-1313.
#
# SURROGATE REFERENCE:
# Schreiber, T. & Schmitz, A. (1996). Improved surrogate data for 
# nonlinearity tests. Physical Review Letters, 77(4), 635.
#
# IHLEN REFERENCE:
# Ihlen, E.A. (2012). Introduction to multifractal detrended fluctuation 
# analysis in Matlab. Frontiers in Physiology, 3, 141.
# 
# Output: D:\Users\admin\Downloads
# ============================================================

import os
import numpy as np
from pathlib import Path
from scipy.fft import fft, ifft
from scipy.stats import linregress, kendalltau
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

logging.getLogger('matplotlib').setLevel(logging.ERROR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from qgis.utils import iface
from qgis.PyQt.QtWidgets import QMessageBox, QInputDialog
from qgis.PyQt.QtCore import QVariant
from qgis.core import NULL

# ============================================================
# OUTPUT DIRECTORY - AS REQUESTED
# ============================================================
OUTPUT_DIR = Path(r"D:\Users\admin\Downloads")

# ============================================================
# CORE DFA FUNCTIONS
# Peng et al. (1994) - original DFA algorithm
# ============================================================

def detrend_segment(segment, order=1):
    """Detrend segment with polynomial of given order"""
    x = np.arange(len(segment))
    coeffs = np.polyfit(x, segment, order)
    trend = np.polyval(coeffs, x)
    return segment - trend

def compute_dfa(ts, scales, order=1, min_scale=8, min_nseg=3):
    """
    Compute DFA exponent alpha following Peng et al. (1994)
    
    PENG DFA ALGORITHM:
    1. Y(k) = sum_{i=1}^{k} [x(i) - <x>]          (profile)
    2. For each scale s:
       a. Divide Y into N_s = floor(N/s) non-overlapping segments
       b. Detrend each segment with polynomial of order m
       c. F^2(s,ν) = (1/s) * sum_{i=1}^{s} [Y_ν(i) - P_ν(i)]^2
    3. F(s) = sqrt[ (1/N_s) * sum_{ν=1}^{N_s} F^2(s,ν) ]
    4. log F(s) = α * log s + const
    5. α from ORDINARY least squares (linregress)
    
    Returns: alpha, intercept, r_value, r2, p_value, std_err, log_s, log_f
    """
    ts = np.asarray(ts, dtype=float)
    ts = ts[np.isfinite(ts)]
    n = len(ts)
    
    if n < 50:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None, None)
    
    # STEP 1: Profile (Peng et al., 1994, Eq. 1)
    profile = np.cumsum(ts - np.mean(ts))
    
    F_vals = []
    S_vals = []
    
    for s in scales:
        if s < min_scale:
            continue
        nseg = n // s
        if nseg < min_nseg:
            continue
        
        # STEP 2: Divide and detrend
        seg = profile[:nseg*s].reshape(nseg, s)
        rms_list = []
        
        for i in range(nseg):
            detrended = detrend_segment(seg[i, :], order)
            rms_list.append(np.sqrt(np.mean(detrended**2)))
        
        F_vals.append(np.mean(rms_list))
        S_vals.append(s)
    
    if len(F_vals) < 5:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None, None)
    
    log_s = np.log10(S_vals)
    log_f = np.log10(F_vals)
    
    # STEP 5: Ordinary least squares (standard DFA regression)
    slope, intercept, r_value, p_value, std_err = linregress(log_s, log_f)
    r2 = r_value**2
    
    return slope, intercept, r_value, r2, p_value, std_err, log_s, log_f

# ============================================================
# MF-DFA (Kantelhardt et al., 2002)
# ============================================================

def correct_multifractal_dfa(ts, scales, order=1, q_range=(-4, 4, 9)):
    """
    CORRECTED MF-DFA: Kantelhardt et al. (2002)
    
    For each q:
    1. F_q(s) = [ (1/N_s) * sum_{ν=1}^{N_s} F^2(s,ν)^{q/2} ]^{1/q}   (q ≠ 0)
    2. F_0(s) = exp[ (1/(2N_s)) * sum_{ν=1}^{N_s} ln(F^2(s,ν)) ]       (q = 0)
    3. log F_q(s) = h(q) * log s + const
    4. h(q) from ORDINARY least squares (linregress) - no weighting
    
    Returns: qs, hq_vals, mf_width, tau_q, alpha_spectrum, f_alpha, is_monofractal, hq_r2_vals
    """
    qs = np.linspace(q_range[0], q_range[1], int(q_range[2]))
    n = len(ts)
    profile = np.cumsum(ts - np.mean(ts))
    
    rms_by_scale = {}
    valid_scales = []
    
    for s in scales:
        if s < 8:
            continue
        nseg = n // s
        if nseg < 4:
            continue
        
        seg = profile[:nseg*s].reshape(nseg, s)
        rms_list = []
        
        for i in range(nseg):
            detrended = detrend_segment(seg[i, :], order)
            rms_val = np.sqrt(np.mean(detrended**2))
            rms_list.append(rms_val)
        
        rms_list = np.array(rms_list)
        rms_list = rms_list[rms_list > 1e-15]
        
        if len(rms_list) > 0:
            rms_by_scale[s] = rms_list
            valid_scales.append(s)
    
    if len(valid_scales) < 5:
        return (np.array([]), np.array([]), np.nan, 
                np.array([]), np.array([]), np.array([]), True, np.array([]))
    
    hq_vals = []
    hq_r2_vals = []
    valid_qs = []
    tau_q = []
    
    for q in qs:
        log_s_vals = []
        log_Fq_vals = []
        
        for s in valid_scales:
            rms_list = rms_by_scale[s]
            
            # Kantelhardt et al. (2002), Eq. 4-5
            if abs(q) < 1e-10:
                # q = 0: logarithmic average
                Fq = np.exp(np.mean(np.log(rms_list)))
            else:
                Fq = (np.mean(rms_list**q))**(1/q)
            
            if Fq > 0 and np.isfinite(Fq):
                log_s_vals.append(np.log10(s))
                log_Fq_vals.append(np.log10(Fq))
        
        if len(log_s_vals) >= 4:
            # Ordinary least squares - same method as DFA (no weights)
            slope, intercept, r_value, p_value, std_err = linregress(
                np.array(log_s_vals), np.array(log_Fq_vals)
            )
            
            if np.isfinite(slope):
                hq_vals.append(slope)
                hq_r2_vals.append(r_value**2)
                valid_qs.append(q)
                tau_q.append(q * slope - 1)
    
    if len(hq_vals) == 0:
        return (np.array([]), np.array([]), np.nan, 
                np.array([]), np.array([]), np.array([]), True, np.array([]))
    
    valid_qs = np.array(valid_qs)
    hq_vals = np.array(hq_vals)
    hq_r2_vals = np.array(hq_r2_vals)
    tau_q = np.array(tau_q)
    
    # Multifractal width (Hölder exponent range)
    multifractal_width = np.max(hq_vals) - np.min(hq_vals)
    is_monofractal = multifractal_width < 0.1
    
    # Singularity spectrum via Legendre transform
    if len(valid_qs) >= 3:
        alpha_spectrum = -np.gradient(tau_q, valid_qs)
        f_alpha = valid_qs * alpha_spectrum + tau_q
        mask = np.isfinite(alpha_spectrum) & np.isfinite(f_alpha)
        alpha_spectrum = alpha_spectrum[mask]
        f_alpha = f_alpha[mask]
    else:
        alpha_spectrum = np.array([])
        f_alpha = np.array([])
    
    return (valid_qs, hq_vals, multifractal_width, 
            tau_q, alpha_spectrum, f_alpha, is_monofractal, hq_r2_vals)

# ============================================================
# SURROGATE AND BOOTSTRAP METHODS
# ============================================================

def iaaft_surrogate(ts, iterations=50):
    """
    IAAFT surrogate: Schreiber & Schmitz (1996)
    Preserves amplitude distribution and power spectrum
    """
    ts = np.asarray(ts)
    sorted_ts = np.sort(ts)
    current = np.random.permutation(ts)
    
    for _ in range(iterations):
        f = fft(current)
        amp = np.abs(f)
        phase = np.exp(1j * np.angle(fft(np.random.permutation(ts))))
        current = np.real(ifft(amp * phase))
        ranks = np.argsort(np.argsort(current))
        current = sorted_ts[ranks]
    
    return current

def stationary_bootstrap(ts, block_size=None):
    """
    Stationary bootstrap: Politis & Romano (1994)
    Resamples blocks of random length with geometric distribution
    PRESERVES dependence/correlation structure
    """
    n = len(ts)
    if block_size is None:
        block_size = max(5, int(np.round(n**(1/3))))
    
    idx = []
    pos = np.random.randint(0, n)
    p_restart = 1.0 / block_size
    
    while len(idx) < n:
        block_len = np.random.geometric(p_restart)
        block_len = min(block_len, n - len(idx))
        
        for j in range(block_len):
            idx.append(pos)
            pos = (pos + 1) % n
        
        if np.random.random() < p_restart:
            pos = np.random.randint(0, n)
    
    return ts[np.array(idx[:n])]

# ============================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================

def check_trend(ts):
    """Check for significant monotonic trend via Kendall τ test"""
    n = len(ts)
    tau, p_trend = kendalltau(np.arange(n), ts)
    return p_trend < 0.05 and abs(tau) > 0.3, tau, p_trend

def check_stationarity(ts):
    """Check stationarity via Augmented Dickey-Fuller test"""
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(ts, autolag='AIC')
        is_stationary = result[1] < 0.05
        return is_stationary, result[1], result[0]
    except ImportError:
        return None, None, None

# ============================================================
# MODULAR ANALYSIS COMPONENTS
# ============================================================

def load_time_series(layer, field_name):
    """Extract numeric time series from QGIS layer with NULL handling"""
    values = []
    for feat in layer.getFeatures():
        val = feat.attribute(field_name)
        if val is None or val == NULL:
            continue
        try:
            values.append(float(val))
        except (ValueError, TypeError):
            continue
    
    ts = np.array(values, dtype=float)
    return ts[np.isfinite(ts)]

def create_scales(ts):
    """Create scale array for DFA with arithmetic spacing"""
    min_scale = max(8, len(ts) // 60)
    max_scale = len(ts) // 5
    if max_scale <= min_scale:
        max_scale = len(ts) // 3
        min_scale = max(4, len(ts) // 100)
    
    step = max(1, (max_scale - min_scale) // 18)
    scales = np.arange(min_scale, max_scale + 1, step)
    if len(scales) < 6:
        scales = np.unique(np.logspace(np.log10(min_scale), 
                                       np.log10(max_scale), 20).astype(int))
    return scales, min_scale, max_scale

def select_dfa_order(has_trend):
    """
    Select detrending order based on trend presence
    DFA-1: sufficient for stationary data (most applications)
    DFA-2: recommended when strong trend is present
    """
    return 2 if has_trend else 1

def run_bootstrap_analysis(ts, scales, order, n_bootstrap=1000):
    """
    Stationary bootstrap for confidence intervals
    Politis & Romano (1994)
    """
    print("Running stationary bootstrap (1000 iterations)...")
    alphas_stat = []
    for i in range(n_bootstrap):
        ts_boot = stationary_bootstrap(ts)
        a, _, _, _, _, _, _, _ = compute_dfa(ts_boot, scales, order=order)
        if not np.isnan(a):
            alphas_stat.append(a)
        if (i + 1) % 250 == 0:
            print(f"  Bootstrap progress: {i+1}/{n_bootstrap}")
    
    # IID bootstrap for comparison
    alphas_iid = []
    for i in range(min(n_bootstrap, 200)):
        ts_boot = np.random.choice(ts, len(ts), replace=True)
        a, _, _, _, _, _, _, _ = compute_dfa(ts_boot, scales, order=order)
        if not np.isnan(a):
            alphas_iid.append(a)
    
    if len(alphas_stat) < 10:
        return None
    
    return {
        'ci_lower': np.percentile(alphas_stat, 2.5),
        'ci_upper': np.percentile(alphas_stat, 97.5),
        'ci_mean': np.mean(alphas_stat),
        'ci_std': np.std(alphas_stat),
        'ci_median': np.median(alphas_stat),
        'n_valid': len(alphas_stat),
        'alphas': alphas_stat,
        'iid_mean': np.mean(alphas_iid) if len(alphas_iid) > 0 else np.nan,
        'iid_std': np.std(alphas_iid) if len(alphas_iid) > 0 else np.nan
    }

def run_surrogate_analysis(ts, scales, order, n_surrogates=200):
    """
    IAAFT surrogate test: Schreiber & Schmitz (1996)
    Null hypothesis: time series is a linear stochastic process
    """
    print("Running surrogate test (200 iterations)...")
    surrogates = []
    for i in range(n_surrogates):
        ts_surr = iaaft_surrogate(ts, iterations=50)
        a, _, _, _, _, _, _, _ = compute_dfa(ts_surr, scales, order=order)
        if not np.isnan(a):
            surrogates.append(a)
        del ts_surr
        if (i + 1) % 50 == 0:
            print(f"  Surrogate progress: {i+1}/{n_surrogates}")
    
    if len(surrogates) == 0:
        return None
    
    return {
        'surrogates': surrogates,
        'surr_mean': np.mean(surrogates),
        'surr_std': np.std(surrogates),
        'surr_median': np.median(surrogates),
        'n_surrogates': len(surrogates)
    }

def compute_p_value(alpha, surr_data):
    """Compute two-sided p-value and standardized effect size (z-score)"""
    surr_array = np.array(surr_data['surrogates'])
    surr_mean = surr_data['surr_mean']
    
    p_val_exact = np.mean(np.abs(surr_array - surr_mean) >= 
                          np.abs(alpha - surr_mean))
    
    if p_val_exact == 0.0:
        p_val = 1.0 / (len(surr_array) + 1)
        p_val_text = f"< {1.0/len(surr_array):.5f}"
    else:
        p_val = p_val_exact
        p_val_text = f"{p_val:.5f}"
    
    # Standardized effect size (z-score)
    effect = (alpha - surr_mean) / surr_data['surr_std'] if surr_data['surr_std'] > 0 else 0.0
    
    return p_val, p_val_text, effect

def run_stability_check(ts, scales, order, n_iter=300):
    """Assess DFA stability via stationary bootstrap"""
    print("Assessing stability (300 iterations)...")
    alphas_stab = []
    for i in range(n_iter):
        ts_boot = stationary_bootstrap(ts)
        a, _, _, _, _, _, _, _ = compute_dfa(ts_boot, scales, order=order)
        if not np.isnan(a):
            alphas_stab.append(a)
    
    if len(alphas_stab) < 2:
        return 0.15, np.nan, np.nan, []
    
    stability = np.std(alphas_stab)
    stability_mean = np.mean(alphas_stab)
    stability_median = np.median(alphas_stab)
    return stability, stability_mean, stability_median, alphas_stab

# ============================================================
# CLASSIFICATION
# ============================================================

def classify_alpha(alpha):
    """Classify Hurst exponent following standard interpretation"""
    if np.isnan(alpha):
        return "UNKNOWN", "unknown"
    
    if alpha > 1.2:
        return "NON-STATIONARY (α > 1.2)", "nonstationary"
    elif alpha > 1.0:
        return "NON-STATIONARY PROCESS", "nonstationary"
    elif alpha > 0.8:
        return "STRONG PERSISTENCE", "persistent"
    elif alpha > 0.65:
        return "MODERATE PERSISTENCE", "persistent"
    elif alpha > 0.55:
        return "WEAK PERSISTENCE", "persistent"
    elif alpha > 0.45:
        return "WHITE NOISE (UNCORRELATED)", "noise"
    elif alpha > 0.35:
        return "WEAK ANTI-PERSISTENCE", "antipersistent"
    else:
        return "STRONG ANTI-PERSISTENCE", "antipersistent"

def classify_multifractal(mf_width):
    """Classify multifractal strength based on Δh width"""
    if np.isnan(mf_width):
        return "UNKNOWN", "unknown"
    
    if mf_width > 0.3:
        return "STRONG MULTIFRACTAL", "multifractal"
    elif mf_width > 0.15:
        return "MODERATE MULTIFRACTAL", "multifractal"
    elif mf_width > 0.1:
        return "WEAK MULTIFRACTAL", "weak_multifractal"
    else:
        return "MONOFRACTAL", "monofractal"

def classify_statistical(p_val, effect):
    """Classify statistical significance"""
    if np.isnan(p_val) or np.isnan(effect):
        return "UNKNOWN", "unknown"
    
    if p_val < 0.01 and abs(effect) > 3.0:
        return "HIGHLY SIGNIFICANT", "highly_significant"
    elif p_val < 0.05 and abs(effect) > 2.0:
        return "SIGNIFICANT", "significant"
    elif p_val < 0.10:
        return "MARGINALLY SIGNIFICANT", "marginally_significant"
    else:
        return "NOT SIGNIFICANT", "not_significant"

def classify_fit(r2):
    """Classify goodness of fit"""
    if np.isnan(r2):
        return "UNKNOWN"
    
    if r2 > 0.95:
        return "EXCELLENT FIT"
    elif r2 > 0.90:
        return "VERY GOOD FIT"
    elif r2 > 0.85:
        return "GOOD FIT"
    elif r2 > 0.75:
        return "ACCEPTABLE FIT"
    else:
        return "POOR FIT"

def determine_status(alpha_category, mf_category, stat_category, 
                     r2, effect, mf_width):
    """Determine final status based on multi-criteria decision matrix"""
    
    if (mf_category in ["multifractal", "weak_multifractal"] and 
        stat_category in ["highly_significant", "significant"] and 
        r2 > 0.85 and abs(effect) > 2.0):
        
        if alpha_category == "nonstationary":
            return "NON-STATIONARY MULTIFRACTAL PROCESS", "darkred"
        else:
            return "STRONG FRACTAL STRUCTURE", "darkgreen"
    
    if (mf_category in ["multifractal", "weak_multifractal"] and 
        stat_category in ["highly_significant", "significant"] and 
        r2 > 0.80):
        return "FRACTAL STRUCTURE", "green"
    
    if (stat_category in ["highly_significant", "significant"] and 
        r2 > 0.85 and abs(effect) > 1.5):
        if mf_category == "multifractal":
            return "SIGNIFICANT MULTIFRACTAL PATTERN", "blue"
        else:
            return "STATISTICALLY SIGNIFICANT STRUCTURE", "teal"
    
    if (alpha_category in ["persistent", "nonstationary"] and 
        r2 > 0.85 and not np.isnan(mf_width) and mf_width > 0.1):
        return "PERSISTENT MULTISCALE STRUCTURE", "darkcyan"
    
    if alpha_category == "nonstationary" and r2 > 0.85:
        return "NON-STATIONARY PROCESS WITH SCALING", "darkorange"
    
    if stat_category in ["highly_significant", "significant"] and r2 > 0.80:
        return "SIGNIFICANT CORRELATION STRUCTURE", "royalblue"
    
    if (alpha_category == "persistent" and r2 > 0.80 and 
        not np.isnan(mf_width) and mf_width > 0.08):
        return "WEAK PERSISTENT STRUCTURE", "limegreen"
    
    if alpha_category == "persistent" and r2 > 0.75:
        return "POSSIBLE WEAK STRUCTURE", "yellowgreen"
    
    if stat_category == "marginally_significant" and r2 > 0.75:
        return "BORDERLINE - REQUIRES MORE DATA", "orange"
    
    if alpha_category == "noise" and stat_category == "not_significant":
        return "WHITE NOISE - NO STRUCTURE DETECTED", "red"
    
    return "INCONCLUSIVE - INSUFFICIENT EVIDENCE", "gray"

def assess_bootstrap_reliability(alpha, ci_mean):
    """Assess reliability of bootstrap estimate"""
    if np.isnan(ci_mean):
        return "UNKNOWN"
    
    if abs(ci_mean - alpha) < 0.1:
        return "HIGH - Consistent with point estimate"
    elif abs(ci_mean - alpha) < 0.2:
        return "MODERATE - Some sampling variation"
    else:
        return "LOW - High uncertainty (check block size)"

# ============================================================
# FIGURE CREATION
# ============================================================

def create_validation_figure(layer_name, field_name, best_order, alpha, 
                             intercept, r2, log_s, log_f, qs, hq_vals, 
                             mf_width, tau_q, alpha_spectrum, f_alpha,
                             surrogates, alphas_stab, ci_mean, surr_mean,
                             p_val_text, effect, ts):
    """Create Figure 1: Validation graphics (6 panels)"""
    
    fig = plt.figure(figsize=(16, 12), dpi=150, facecolor='white')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # Panel A: DFA Scaling
    ax1 = fig.add_subplot(gs[0, 0])
    if log_s is not None and len(log_s) > 0:
        ax1.plot(log_s, log_f, 'ko-', markersize=7, linewidth=1.5, 
                label='Data', zorder=5)
        fit_line = alpha * log_s + intercept
        ax1.plot(log_s, fit_line, 'r-', linewidth=2.5, 
                label=f'α = {alpha:.4f}', zorder=4)
        ax1.set_xlabel('log₁₀(Scale)', fontsize=11)
        ax1.set_ylabel('log₁₀(F(s))', fontsize=11)
        ax1.set_title(f'A: DFA Scaling (DFA-{best_order}, R² = {r2:.4f})', 
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Panel B: h(q) Spectrum
    ax2 = fig.add_subplot(gs[0, 1])
    if len(qs) > 0 and len(hq_vals) > 0:
        ax2.plot(qs, hq_vals, 'bo-', linewidth=2, markersize=7, label='h(q)')
        ax2.axhline(y=alpha, color='r', linestyle='--', alpha=0.7, 
                   linewidth=2, label=f'α = {alpha:.3f}')
        ax2.fill_between(qs, np.min(hq_vals), np.max(hq_vals), 
                        alpha=0.15, color='blue')
        ax2.set_xlabel('Moment q', fontsize=11)
        ax2.set_ylabel('Hölder exponent h(q)', fontsize=11)
        ax2.set_title(f'B: MF-DFA Spectrum (Δh = {mf_width:.4f})', 
                     fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
    else:
        ax2.text(0.5, 0.5, 'Insufficient data\nfor MF-DFA', 
                ha='center', va='center', transform=ax2.transAxes, 
                fontsize=14, color='gray')
        ax2.set_title('B: MF-DFA Spectrum', fontsize=12, fontweight='bold')
    
    # Panel C: τ(q) - Mass Exponent
    ax3 = fig.add_subplot(gs[0, 2])
    if len(qs) > 0 and len(tau_q) > 0:
        ax3.plot(qs, tau_q, 'go-', linewidth=2, markersize=7, label='τ(q)')
        tau_mono = qs * alpha - 1
        ax3.plot(qs, tau_mono, 'r--', linewidth=2, alpha=0.7, 
                label=f'Monofractal (α={alpha:.3f})')
        ax3.set_xlabel('Moment q', fontsize=11)
        ax3.set_ylabel('Mass exponent τ(q)', fontsize=11)
        ax3.set_title('C: Mass Exponent τ(q)', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax3.transAxes, fontsize=14, color='gray')
        ax3.set_title('C: Mass Exponent τ(q)', fontsize=12, fontweight='bold')
    
    # Panel D: Singularity Spectrum f(α)
    ax4 = fig.add_subplot(gs[1, 0])
    if len(alpha_spectrum) > 0 and len(f_alpha) > 0:
        ax4.plot(alpha_spectrum, f_alpha, 'mo-', linewidth=2, markersize=7)
        ax4.set_xlabel('Singularity strength α', fontsize=11)
        ax4.set_ylabel('Singularity spectrum f(α)', fontsize=11)
        ax4.set_title(f'D: Singularity Spectrum (width = {mf_width:.3f})', 
                     fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax4.transAxes, fontsize=14, color='gray')
        ax4.set_title('D: Singularity Spectrum', fontsize=12, fontweight='bold')
    
    # Panel E: Bootstrap & Surrogate Distributions
    ax5 = fig.add_subplot(gs[1, 1])
    if len(surrogates) > 0 and len(alphas_stab) > 0:
        ax5.hist(alphas_stab, bins=30, alpha=0.5, color='forestgreen', 
                edgecolor='black', linewidth=0.5, 
                label=f'Stationary bootstrap (n={len(alphas_stab)})', 
                density=True)
        ax5.hist(surrogates, bins=30, alpha=0.5, color='cornflowerblue', 
                edgecolor='black', linewidth=0.5, 
                label=f'Surrogates (n={len(surrogates)})', density=True)
        ax5.axvline(alpha, color='red', linewidth=2.5, 
                   label=f'Observed α = {alpha:.3f}')
        ax5.axvline(ci_mean, color='darkgreen', linestyle='--', linewidth=2,
                   label=f'Bootstrap mean = {ci_mean:.3f}')
        ax5.axvline(surr_mean, color='blue', linestyle='--', linewidth=2,
                   label=f'Surrogate mean = {surr_mean:.3f}')
        ax5.set_xlabel('α (Hurst exponent)', fontsize=11)
        ax5.set_ylabel('Probability Density', fontsize=11)
        ax5.set_title(f'E: Distributions\np = {p_val_text}, z-score = {effect:.2f}',
                     fontsize=12, fontweight='bold')
        ax5.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
    else:
        ax5.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax5.transAxes, fontsize=14, color='gray')
        ax5.set_title('E: Distributions', fontsize=12, fontweight='bold')
    
    # Panel F: Time Series Preview
    ax6 = fig.add_subplot(gs[1, 2])
    ts_plot = ts[:min(500, len(ts))]
    ax6.plot(np.arange(len(ts_plot)), ts_plot, color='dimgray', 
            linewidth=0.8, alpha=0.8)
    ax6.set_xlabel('Index', fontsize=10)
    ax6.set_ylabel('Value', fontsize=10)
    ax6.set_title(f'F: Time Series Preview ({len(ts_plot)} of {len(ts)} points)', 
                 fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle(f'DFA/MF-DFA Validation Graphics: {layer_name} - {field_name}',
                fontsize=15, fontweight='bold', color='navy', y=1.01)
    
    return fig

def create_summary_figure(layer_name, field_name, n_points, best_order,
                          min_scale, max_scale, n_scales, alpha, 
                          ci_lower, ci_upper, ci_mean, stability, r2,
                          alpha_class, mf_class, stat_class, fit_class,
                          mf_width, p_val_text, effect, bootstrap_reliability,
                          confidence, bootstrap_bias, surr_mean, surr_std,
                          iid_mean, status):
    """Create Figure 2: Results Summary (clean, large text)"""
    
    fig = plt.figure(figsize=(12, 14), dpi=150, facecolor='white')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    ci_text = f"[{ci_lower:.4f}, {ci_upper:.4f}]" if not np.isnan(ci_lower) else "N/A"
    mf_width_text = f"{mf_width:.4f}" if not np.isnan(mf_width) else "N/A"
    iid_text = f"{iid_mean:.4f}" if not np.isnan(iid_mean) else "N/A"
    surr_mean_text = f"{surr_mean:.4f}" if not np.isnan(surr_mean) else "N/A"
    
    summary_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    DFA/MF-DFA ANALYSIS RESULTS                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Layer  : {layer_name:<52}║
║  Field  : {field_name:<52}║
║  Points : {n_points:<6}  |  Order : DFA-{best_order:<3}  |  Scales : {min_scale}-{max_scale} ({n_scales} pts)        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  PRIMARY METRICS:                                                ║
║  ─────────────────────────────────────────────────────────────  ║
║  α (Hurst Exponent)          = {alpha:>8.4f}                     ║
║  Stationary Bootstrap 95% CI = {ci_text:<26}               ║
║  Stationary Bootstrap Mean   = {ci_mean if not np.isnan(ci_mean) else 'N/A':>8.4f}                     ║
║  Bootstrap Stability (σ)     = {stability:>8.4f}                     ║
║  R² (Goodness of Fit)        = {r2:>8.4f}                     ║
║                                                                  ║
║  CLASSIFICATION:                                                 ║
║  ─────────────────────────────────────────────────────────────  ║
║  Hurst Classification    = {alpha_class:<34}║
║  Multifractal Class      = {mf_class:<34}║
║  Statistical Class       = {stat_class:<34}║
║  Fit Quality             = {fit_class:<34}║
║                                                                  ║
║  ADVANCED METRICS:                                               ║
║  ─────────────────────────────────────────────────────────────  ║
║  Multifractal Width (Δh)   = {mf_width_text:<27}               ║
║  P-value (Two-sided)       = {p_val_text:<27}               ║
║  Standardized Effect Size  = {effect:>8.2f}                     ║
║  Bootstrap Reliability     = {bootstrap_reliability:<34}║
║  Informational Confidence  = {confidence:>7.1%}                      ║
║  Bootstrap Bias (α-μ_boot) = {bootstrap_bias:>8.4f}                     ║
║                                                                  ║
║  REFERENCE VALUES:                                               ║
║  ─────────────────────────────────────────────────────────────  ║
║  IAAFT Surrogate Mean α    = {surr_mean_text:<27}               ║
║  IAAFT Surrogate Std α     = {surr_std if not np.isnan(surr_std) else 'N/A':>8.4f}                     ║
║  IID Bootstrap Mean α      = {iid_text:<27}               ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  FINAL STATUS: {status:<42}║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

INTERPRETATION GUIDE:
• α = 0.5 → White noise (no correlation)
• α > 0.5 → Long-range persistence/correlation
• α < 0.5 → Anti-persistence (mean-reverting)
• α > 1.0 → Non-stationary process
• Δh > 0.15 → Multifractal behavior
• p < 0.05 → Statistically significant vs. random surrogates
• Stationary bootstrap CI → Valid for dependent data (CORRECT)
• IID bootstrap → SHOWN FOR COMPARISON ONLY (biased)

REFERENCES:
[1] Peng et al. (1994) Phys. Rev. E 49:1685-1689 (Original DFA)
[2] Kantelhardt et al. (2002) Physica A 316:87-114 (MF-DFA)
[3] Politis & Romano (1994) JASA 89:1303-1313 (Stationary bootstrap)
[4] Schreiber & Schmitz (1996) PRL 77:635 (IAAFT surrogates)
"""
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center', 
            horizontalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', 
                     alpha=0.95, pad=1.5))
    
    fig.suptitle(f'DFA/MF-DFA Results: {layer_name} - {field_name}',
                fontsize=16, fontweight='bold', color='navy', y=0.99)
    
    return fig

# ============================================================
# DETAILED EXCEL REPORT EXPORT
# ============================================================

def export_detailed_excel_report(output_dir, safe_name, results_dict, 
                                 log_s, log_f, qs, hq_vals, hq_r2_vals,
                                 tau_q, alpha_spectrum, f_alpha,
                                 surrogates, alphas_stab, ts):
    """
    Export comprehensive Excel report with multiple sheets:
    1. Summary - Main results and classifications
    2. DFA_Scaling - log(s) vs log(F) data points
    3. MFDFA_hq - q vs h(q) with R² values
    4. Mass_Exponent_Tau - q vs τ(q) mass exponent
    5. Singularity_Spectrum - α vs f(α)
    6. Bootstrap_Distribution - Stationary bootstrap α values
    7. Surrogate_Distribution - IAAFT surrogate α values
    8. Raw_Time_Series - The original time series data
    """
    
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        print("openpyxl not available. Install with: pip install openpyxl")
        print("Falling back to CSV export only...")
        export_results_csv(output_dir, safe_name, results_dict)
        return False
    
    try:
        wb = openpyxl.Workbook()
        
        # Define styles
        header_font = Font(name='Calibri', size=12, bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='2F5496', end_color='2F5496', fill_type='solid')
        subheader_font = Font(name='Calibri', size=11, bold=True, color='1F3864')
        subheader_fill = PatternFill(start_color='D6E4F0', end_color='D6E4F0', fill_type='solid')
        data_font = Font(name='Calibri', size=11)
        status_font = Font(name='Calibri', size=12, bold=True, color='C00000')
        title_font = Font(name='Calibri', size=14, bold=True, color='1F3864')
        center_align = Alignment(horizontal='center', vertical='center')
        
        # =====================================================
        # SHEET 1: SUMMARY
        # =====================================================
        ws1 = wb.active
        ws1.title = "Summary"
        
        # Title
        ws1.merge_cells('A1:D1')
        ws1['A1'] = "DFA/MF-DFA Analysis Report"
        ws1['A1'].font = title_font
        ws1['A1'].alignment = Alignment(horizontal='center')
        
        ws1.merge_cells('A2:D2')
        ws1['A2'] = f"Layer: {results_dict['layer']} | Field: {results_dict['field']}"
        ws1['A2'].font = Font(name='Calibri', size=11, italic=True)
        ws1['A2'].alignment = Alignment(horizontal='center')
        
        ws1.merge_cells('A3:D3')
        ws1['A3'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ws1['A3'].font = Font(name='Calibri', size=10, italic=True, color='666666')
        ws1['A3'].alignment = Alignment(horizontal='center')
        
        # Helper function for sections
        def write_section(ws, start_row, title, data_pairs):
            ws.merge_cells(f'A{start_row}:D{start_row}')
            ws[f'A{start_row}'] = title
            ws[f'A{start_row}'].font = header_font
            ws[f'A{start_row}'].fill = header_fill
            ws[f'A{start_row}'].alignment = center_align
            
            for i, (label, value) in enumerate(data_pairs):
                r = start_row + 1 + i
                ws[f'A{r}'] = label
                ws[f'A{r}'].font = subheader_font
                ws[f'A{r}'].fill = subheader_fill
                ws[f'B{r}'] = str(value) if not isinstance(value, str) else value
                ws[f'B{r}'].font = data_font
                ws.merge_cells(f'B{r}:D{r}')
            
            return start_row + len(data_pairs) + 2
        
        row = 5
        
        # Input Data
        row = write_section(ws1, row, "INPUT DATA", [
            ("Layer Name", results_dict['layer']),
            ("Field Name", results_dict['field']),
            ("Number of Points", results_dict['n_points']),
            ("Detrending Order", f"DFA-{results_dict['detrending_order']}"),
            ("Scale Range", f"{results_dict.get('scale_min', 'N/A')} - {results_dict.get('scale_max', 'N/A')}"),
            ("Number of Scales", results_dict.get('n_scales', 'N/A')),
        ])
        
        # Diagnostics
        row = write_section(ws1, row, "DIAGNOSTICS", [
            ("Kendall τ", f"{results_dict.get('kendall_tau', 'N/A'):.4f}" if results_dict.get('kendall_tau') is not None else 'N/A'),
            ("Kendall p-value", f"{results_dict.get('kendall_p', 'N/A'):.4f}" if results_dict.get('kendall_p') is not None else 'N/A'),
            ("Trend Detected", "Yes" if results_dict.get('has_trend') else "No"),
            ("ADF Stationarity", "Stationary" if results_dict.get('is_stationary_adf') else ("Non-stationary" if results_dict.get('is_stationary_adf') is not None else 'N/A')),
            ("ADF p-value", f"{results_dict.get('adf_pvalue', 'N/A'):.4f}" if results_dict.get('adf_pvalue') is not None else 'N/A'),
        ])
        
        # DFA Results
        row = write_section(ws1, row, "PRIMARY DFA RESULTS", [
            ("α (Hurst Exponent)", f"{results_dict['alpha']:.6f}"),
            ("Intercept", f"{results_dict['intercept']:.6f}"),
            ("R (Correlation Coefficient)", f"{results_dict.get('r_value', 'N/A'):.6f}" if results_dict.get('r_value') is not None else 'N/A'),
            ("R² (Goodness of Fit)", f"{results_dict['r2']:.6f}"),
            ("Regression p-value", f"{results_dict.get('reg_p_value', 'N/A'):.6e}" if results_dict.get('reg_p_value') is not None else 'N/A'),
            ("Standard Error of α", f"{results_dict.get('reg_std_err', 'N/A'):.6f}" if results_dict.get('reg_std_err') is not None else 'N/A'),
            ("Fit Quality", results_dict['fit_quality']),
        ])
        
        # Bootstrap Results
        row = write_section(ws1, row, "STATIONARY BOOTSTRAP (Politis & Romano, 1994)", [
            ("95% CI Lower", f"{results_dict['ci_lower_stationary']:.6f}" if not np.isnan(results_dict['ci_lower_stationary']) else 'N/A'),
            ("95% CI Upper", f"{results_dict['ci_upper_stationary']:.6f}" if not np.isnan(results_dict['ci_upper_stationary']) else 'N/A'),
            ("Bootstrap Mean", f"{results_dict['ci_mean_stationary']:.6f}" if not np.isnan(results_dict['ci_mean_stationary']) else 'N/A'),
            ("Bootstrap Std Dev", f"{results_dict['ci_std_stationary']:.6f}" if not np.isnan(results_dict['ci_std_stationary']) else 'N/A'),
            ("Bootstrap Bias (α - μ_boot)", f"{results_dict['bootstrap_bias']:.6f}" if not np.isnan(results_dict['bootstrap_bias']) else 'N/A'),
            ("Bootstrap Reliability", results_dict['bootstrap_reliability']),
            ("IID Bootstrap Mean (comparison)", f"{results_dict['iid_bootstrap_mean']:.6f}" if not np.isnan(results_dict['iid_bootstrap_mean']) else 'N/A'),
            ("IID Bootstrap Std (comparison)", f"{results_dict['iid_bootstrap_std']:.6f}" if not np.isnan(results_dict['iid_bootstrap_std']) else 'N/A'),
        ])
        
        # Surrogate Test
        row = write_section(ws1, row, "IAAFT SURROGATE TEST (Schreiber & Schmitz, 1996)", [
            ("Surrogate Mean α", f"{results_dict['surrogate_mean']:.6f}" if not np.isnan(results_dict['surrogate_mean']) else 'N/A'),
            ("Surrogate Std α", f"{results_dict['surrogate_std']:.6f}" if not np.isnan(results_dict['surrogate_std']) else 'N/A'),
            ("Number of Surrogates", results_dict['n_surrogates']),
            ("P-value (two-sided)", results_dict['p_value_text']),
            ("P-value (numeric)", f"{results_dict['p_value_two_sided']:.6e}" if not np.isnan(results_dict['p_value_two_sided']) else 'N/A'),
            ("Standardized Effect Size (z-score)", f"{results_dict['effect_size_z_score']:.3f}"),
        ])
        
        # MF-DFA Results
        row = write_section(ws1, row, "MULTIFRACTAL DFA (Kantelhardt et al., 2002)", [
            ("Multifractal Width (Δh)", f"{results_dict['mf_width_delta_h']:.6f}" if not np.isnan(results_dict['mf_width_delta_h']) else 'N/A'),
            ("Singularity Range (Δα)", f"{results_dict['delta_alpha']:.6f}" if not np.isnan(results_dict.get('delta_alpha', np.nan)) else 'N/A'),
            ("Is Monofractal", "Yes" if results_dict['is_monofractal'] else "No"),
            ("MF Classification", results_dict['mf_class']),
        ])
        
        # Classification
        row = write_section(ws1, row, "CLASSIFICATION SUMMARY", [
            ("Hurst Classification", results_dict['alpha_class']),
            ("Multifractal Classification", results_dict['mf_class']),
            ("Statistical Classification", results_dict['stat_class']),
            ("Fit Quality", results_dict['fit_quality']),
            ("Informational Confidence", f"{results_dict['confidence_informational']:.2%}"),
        ])
        
        # Final Status
        ws1.merge_cells(f'A{row}:D{row}')
        ws1[f'A{row}'] = "FINAL STATUS"
        ws1[f'A{row}'].font = Font(name='Calibri', size=14, bold=True, color='FFFFFF')
        ws1[f'A{row}'].fill = PatternFill(start_color='C00000', end_color='C00000', fill_type='solid')
        ws1[f'A{row}'].alignment = center_align
        
        ws1.merge_cells(f'A{row+1}:D{row+1}')
        ws1[f'A{row+1}'] = results_dict['status']
        ws1[f'A{row+1}'].font = status_font
        ws1[f'A{row+1}'].alignment = center_align
        
        # References
        row += 3
        ws1.merge_cells(f'A{row}:D{row}')
        ws1[f'A{row}'] = "REFERENCES"
        ws1[f'A{row}'].font = header_font
        ws1[f'A{row}'].fill = header_fill
        
        refs = [
            "[1] Peng, C.K. et al. (1994). Phys. Rev. E 49(2):1685-1689.",
            "[2] Kantelhardt, J.W. et al. (2002). Physica A 316(1-4):87-114.",
            "[3] Politis, D.N. & Romano, J.P. (1994). JASA 89(428):1303-1313.",
            "[4] Schreiber, T. & Schmitz, A. (1996). PRL 77(4):635.",
        ]
        for i, ref in enumerate(refs):
            ws1.merge_cells(f'A{row+1+i}:D{row+1+i}')
            ws1[f'A{row+1+i}'] = ref
            ws1[f'A{row+1+i}'].font = Font(name='Calibri', size=10, italic=True)
        
        ws1.column_dimensions['A'].width = 30
        ws1.column_dimensions['B'].width = 25
        ws1.column_dimensions['C'].width = 20
        ws1.column_dimensions['D'].width = 20
        
        # =====================================================
        # SHEETS 2-8: DATA SHEETS
        # =====================================================
        def write_data_sheet(wb, sheet_name, headers, data_func, col_widths=None):
            ws = wb.create_sheet(sheet_name)
            for j, header in enumerate(headers):
                col_letter = chr(65 + j) if j < 26 else chr(64 + j // 26) + chr(65 + j % 26)
                ws[f'{col_letter}1'] = header
                ws[f'{col_letter}1'].font = header_font
                ws[f'{col_letter}1'].fill = header_fill
                ws[f'{col_letter}1'].alignment = center_align
            data_func(ws)
            if col_widths:
                for j, width in enumerate(col_widths):
                    col_letter = chr(65 + j) if j < 26 else chr(64 + j // 26) + chr(65 + j % 26)
                    ws.column_dimensions[col_letter].width = width
        
        # Sheet 2: DFA Scaling
        def dfa_scaling_data(ws):
            if log_s is not None and log_f is not None:
                for i in range(len(log_s)):
                    ws[f'A{i+2}'] = log_s[i]
                    ws[f'B{i+2}'] = 10**log_s[i]
                    ws[f'C{i+2}'] = log_f[i]
                    ws[f'D{i+2}'] = 10**log_f[i]
                    ws[f'E{i+2}'] = results_dict['alpha'] * log_s[i] + results_dict['intercept']
        
        write_data_sheet(wb, "DFA_Scaling", 
                        ['log10(Scale)', 'Scale (s)', 'log10(F(s))', 'F(s)', 'Fitted log10(F)'],
                        dfa_scaling_data, [18, 18, 18, 18, 18])
        
        # Sheet 3: MF-DFA h(q)
        def mfdfa_hq_data(ws):
            if len(qs) > 0 and len(hq_vals) > 0:
                for i in range(len(qs)):
                    ws[f'A{i+2}'] = qs[i]
                    ws[f'B{i+2}'] = hq_vals[i]
                    if len(hq_r2_vals) > i:
                        ws[f'C{i+2}'] = hq_r2_vals[i]
        
        write_data_sheet(wb, "MFDFA_hq",
                        ['q', 'h(q)', 'R² of h(q) fit'],
                        mfdfa_hq_data, [18, 18, 18])
        
        # Sheet 4: Mass Exponent τ(q)
        def mass_exponent_data(ws):
            if len(qs) > 0 and len(tau_q) > 0:
                for i in range(len(qs)):
                    ws[f'A{i+2}'] = qs[i]
                    ws[f'B{i+2}'] = tau_q[i]
                    ws[f'C{i+2}'] = qs[i] * results_dict['alpha'] - 1
        
        write_data_sheet(wb, "Mass_Exponent_Tau",
                        ['q', 'τ(q)', 'τ_monofractal (α*q - 1)'],
                        mass_exponent_data, [20, 20, 20])
        
        # Sheet 5: Singularity Spectrum
        def singularity_data(ws):
            if len(alpha_spectrum) > 0 and len(f_alpha) > 0:
                for i in range(len(alpha_spectrum)):
                    ws[f'A{i+2}'] = alpha_spectrum[i]
                    ws[f'B{i+2}'] = f_alpha[i]
        
        write_data_sheet(wb, "Singularity_Spectrum",
                        ['α (Singularity strength)', 'f(α) (Singularity spectrum)'],
                        singularity_data, [25, 25])
        
        # Sheet 6: Bootstrap Distribution
        def bootstrap_data(ws):
            if len(alphas_stab) > 0:
                for i, val in enumerate(alphas_stab):
                    ws[f'A{i+2}'] = val
        
        write_data_sheet(wb, "Bootstrap_Distribution",
                        ['Stationary Bootstrap α values'],
                        bootstrap_data, [25])
        
        # Sheet 7: Surrogate Distribution
        def surrogate_data(ws):
            if len(surrogates) > 0:
                for i, val in enumerate(surrogates):
                    ws[f'A{i+2}'] = val
        
        write_data_sheet(wb, "Surrogate_Distribution",
                        ['IAAFT Surrogate α values'],
                        surrogate_data, [25])
        
        # Sheet 8: Raw Time Series
        def raw_ts_data(ws):
            for i, val in enumerate(ts):
                ws[f'A{i+2}'] = i + 1
                ws[f'B{i+2}'] = val
        
        write_data_sheet(wb, "Raw_Time_Series",
                        ['Index', 'Value'],
                        raw_ts_data, [18, 18])
        
        # Save
        excel_path = output_dir / f"DFA_Report_{safe_name[:50]}.xlsx"
        wb.save(str(excel_path))
        print(f"Detailed Excel report saved to: {excel_path}")
        return True
        
    except (OSError, ValueError, KeyError) as e:
        print(f"Error creating Excel report: {e}")
        export_results_csv(output_dir, safe_name, results_dict)
        return False


def export_results_csv(output_dir, safe_name, results_dict):
    """Fallback CSV export"""
    try:
        import pandas as pd
        results = results_dict.copy()
        results['timestamp'] = datetime.now().isoformat()
        df = pd.DataFrame([results])
        csv_path = output_dir / f"DFA_results_{safe_name[:50]}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to CSV: {csv_path}")
        return True
    except ImportError:
        print("pandas not available - cannot save CSV")
        return False
    except (OSError, ValueError) as e:
        print(f"Cannot write CSV: {e}")
        return False

# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def run_dfa_analysis():
    """Main DFA analysis pipeline"""
    
    # Get active layer
    layer = iface.activeLayer()
    if not layer:
        QMessageBox.critical(None, "Error", 
                           "No active layer! Please select a vector layer.")
        return
    
    # Find numeric fields
    numeric_fields = []
    for f in layer.fields():
        if f.type() in [QVariant.Int, QVariant.Double]:
            numeric_fields.append(f.name())
    
    if not numeric_fields:
        QMessageBox.critical(None, "Error", 
                           "No numeric fields found in the active layer!")
        return
    
    # Select field
    field_name, ok = QInputDialog.getItem(None, "DFA Analysis", 
                                          "Select numeric field:", 
                                          numeric_fields, 0, False)
    if not ok:
        return
    
    # Ensure output directory exists
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        QMessageBox.critical(None, "Error", 
                           f"Cannot access output directory:\n{OUTPUT_DIR}\n\nError: {e}")
        return
    
    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    print("\n" + "="*70)
    print(f"DFA/MF-DFA ANALYSIS - {layer.name()} / {field_name}")
    print("="*70)
    print(f"Method: Peng et al. (1994) DFA + Kantelhardt et al. (2002) MF-DFA")
    print(f"Output directory: {OUTPUT_DIR}")
    
    ts = load_time_series(layer, field_name)
    
    if len(ts) < 50:
        QMessageBox.warning(None, "Error", 
                          f"Insufficient data! Found {len(ts)} points (need 50+)")
        return
    
    print(f"Points: {len(ts)}")
    
    # ============================================================
    # 2. DIAGNOSTICS
    # ============================================================
    has_trend, tau_val, p_trend = check_trend(ts)
    is_stationary, adf_pvalue, adf_stat = check_stationarity(ts)
    
    if has_trend:
        print(f"Strong trend detected (Kendall τ = {tau_val:.3f}, p = {p_trend:.4f})")
    if is_stationary is not None:
        print(f"Stationarity (ADF p-value): {adf_pvalue:.4f} - "
              f"{'Stationary' if is_stationary else 'Non-stationary'}")
    
    # ============================================================
    # 3. CREATE SCALES
    # ============================================================
    scales, min_scale, max_scale = create_scales(ts)
    print(f"Scales: {min_scale} to {max_scale} ({len(scales)} steps)")
    
    # ============================================================
    # 4. SELECT DFA ORDER
    # ============================================================
    best_order = select_dfa_order(has_trend)
    print(f"Detrending order: DFA-{best_order} "
          f"({'trend detected' if has_trend else 'stationary/no trend'})")
    
    # ============================================================
    # 5. COMPUTE DFA
    # ============================================================
    print("Computing DFA...")
    alpha, intercept, r_value, r2, reg_p_value, reg_std_err, log_s, log_f = \
        compute_dfa(ts, scales, order=best_order)
    
    if np.isnan(alpha):
        QMessageBox.warning(None, "Error", "DFA analysis failed!")
        return
    
    if len(ts) < 500:
        print(f"  Note: Short series ({len(ts)} points) - results may have higher uncertainty")
    
    print(f"  α (Hurst exponent) = {alpha:.4f}")
    print(f"  Intercept = {intercept:.4f}")
    print(f"  R = {r_value:.4f}")
    print(f"  R² = {r2:.4f}")
    print(f"  Regression p-value = {reg_p_value:.6e}")
    print(f"  Standard error = {reg_std_err:.6f}")
    
    # ============================================================
    # 6. BOOTSTRAP ANALYSIS
    # ============================================================
    boot_results = run_bootstrap_analysis(ts, scales, best_order)
    
    if boot_results is None:
        print("  Bootstrap analysis failed")
        ci_lower = ci_upper = ci_mean = ci_std = ci_median = np.nan
        iid_mean = iid_std = np.nan
        alphas_stab = []
    else:
        ci_lower = boot_results['ci_lower']
        ci_upper = boot_results['ci_upper']
        ci_mean = boot_results['ci_mean']
        ci_std = boot_results['ci_std']
        ci_median = boot_results['ci_median']
        iid_mean = boot_results['iid_mean']
        iid_std = boot_results['iid_std']
        alphas_stab = boot_results['alphas']
        
        print(f"  Stationary bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Stationary bootstrap mean = {ci_mean:.4f}, median = {ci_median:.4f}")
        print(f"  Stationary bootstrap std = {ci_std:.4f}")
        if not np.isnan(iid_mean):
            print(f"  IID bootstrap mean = {iid_mean:.4f} (comparison only)")
    
    # ============================================================
    # 7. STABILITY CHECK
    # ============================================================
    stability, stability_mean, stability_median, _ = run_stability_check(ts, scales, best_order)
    print(f"  Stability (std) = {stability:.4f}")
    
    # ============================================================
    # 8. SURROGATE ANALYSIS
    # ============================================================
    surr_data = run_surrogate_analysis(ts, scales, best_order)
    
    if surr_data is None:
        print("  Surrogate analysis failed")
        p_val, p_val_text, effect = 1.0, "1.00000", 0.0
        surr_mean = surr_std = surr_median = np.nan
        surrogates = []
    else:
        surr_mean = surr_data['surr_mean']
        surr_std = surr_data['surr_std']
        surr_median = surr_data['surr_median']
        surrogates = surr_data['surrogates']
        p_val, p_val_text, effect = compute_p_value(alpha, surr_data)
        
        print(f"  Surrogate mean α = {surr_mean:.4f}, median = {surr_median:.4f}")
        print(f"  Surrogate std α = {surr_std:.4f}")
        print(f"  P-value (two-sided) = {p_val_text}")
        print(f"  Standardized effect size (z-score) = {effect:.3f}")
    
    # ============================================================
    # 9. MULTIFRACTAL ANALYSIS
    # ============================================================
    print("Multifractal analysis (Kantelhardt et al., 2002)...")
    mf_result = correct_multifractal_dfa(ts, scales, order=best_order)
    
    qs, hq_vals, mf_width, tau_q, alpha_spectrum, f_alpha, is_monofractal, hq_r2_vals = mf_result
    
    if len(qs) > 0:
        delta_h = mf_width
        delta_alpha = (np.max(alpha_spectrum) - np.min(alpha_spectrum) 
                      if len(alpha_spectrum) > 0 else np.nan)
        
        print(f"  Multifractal width Δh = {delta_h:.4f}")
        if not np.isnan(delta_alpha):
            print(f"  Singularity strength range Δα = {delta_alpha:.4f}")
        print(f"  Monofractal behavior: {'YES (Δh < 0.1)' if is_monofractal else 'NO'}")
    else:
        print("  Insufficient data for MF-DFA")
        delta_h = np.nan
        delta_alpha = np.nan
    
    # ============================================================
    # 10. CLASSIFICATION
    # ============================================================
    alpha_class, alpha_category = classify_alpha(alpha)
    mf_class, mf_category = classify_multifractal(mf_width)
    stat_class, stat_category = classify_statistical(p_val, effect)
    fit_class = classify_fit(r2)
    bootstrap_reliability = assess_bootstrap_reliability(alpha, ci_mean)
    bootstrap_bias = alpha - ci_mean if not np.isnan(ci_mean) else np.nan
    status, color = determine_status(alpha_category, mf_category, 
                                     stat_category, r2, effect, mf_width)
    
    # Informational confidence (for UI reference only)
    r2_score = max(0, min(1, (r2 - 0.80) / 0.20)) if not np.isnan(r2) else 0
    alpha_score = 1 - min(1, abs(alpha - 0.5) / 0.5)
    stab_score = 1 - min(1, stability / 0.20)
    mf_score = 1 - min(1, mf_width / 0.4) if not np.isnan(mf_width) else 0.5
    stat_score = 1 - min(p_val, 1.0) if not np.isnan(p_val) else 0
    confidence = (0.20 * r2_score + 0.20 * alpha_score + 
                  0.15 * stab_score + 0.20 * mf_score + 0.25 * stat_score)
    
    print("-"*70)
    print("ANALYSIS RESULTS:")
    print(f"  DFA-{best_order}: α = {alpha:.4f}, R² = {r2:.4f}")
    print(f"  Classification: {alpha_class}")
    if not np.isnan(mf_width):
        print(f"  Multifractal: {mf_class} (Δh = {mf_width:.4f}, Δα = {delta_alpha:.4f})")
    else:
        print(f"  Multifractal: {mf_class}")
    print(f"  Statistical: {stat_class} (p = {p_val_text}, z-score = {effect:.2f})")
    print(f"  Fit quality: {fit_class}")
    print(f"  Bootstrap reliability: {bootstrap_reliability}")
    print(f"  Bootstrap bias: {bootstrap_bias:.4f}")
    print(f"  FINAL STATUS: {status}")
    print("="*70)
    
    # ============================================================
    # 11. PREPARE RESULTS DICTIONARY
    # ============================================================
    safe_name = (f"{layer.name()}_{field_name}"
                 .replace("/", "_").replace("\\", "_")
                 .replace(" ", "_").replace(":", ""))
    
    results_dict = {
        'layer': layer.name(),
        'field': field_name,
        'n_points': len(ts),
        'detrending_order': best_order,
        'scale_min': min_scale,
        'scale_max': max_scale,
        'n_scales': len(scales),
        'alpha': alpha,
        'intercept': intercept,
        'r_value': r_value,
        'r2': r2,
        'reg_p_value': reg_p_value,
        'reg_std_err': reg_std_err,
        'ci_lower_stationary': ci_lower if not np.isnan(ci_lower) else np.nan,
        'ci_upper_stationary': ci_upper if not np.isnan(ci_upper) else np.nan,
        'ci_mean_stationary': ci_mean if not np.isnan(ci_mean) else np.nan,
        'ci_std_stationary': ci_std if not np.isnan(ci_std) else np.nan,
        'ci_median_stationary': ci_median if not np.isnan(ci_median) else np.nan,
        'bootstrap_bias': bootstrap_bias,
        'iid_bootstrap_mean': iid_mean if not np.isnan(iid_mean) else np.nan,
        'iid_bootstrap_std': iid_std if not np.isnan(iid_std) else np.nan,
        'fit_quality': fit_class,
        'stability': stability,
        'stability_mean': stability_mean,
        'stability_median': stability_median,
        'bootstrap_reliability': bootstrap_reliability,
        'mf_width_delta_h': delta_h,
        'delta_alpha': delta_alpha if not np.isnan(delta_alpha) else np.nan,
        'is_monofractal': is_monofractal,
        'mf_class': mf_class,
        'p_value_two_sided': p_val,
        'p_value_text': p_val_text,
        'effect_size_z_score': effect,
        'stat_class': stat_class,
        'surrogate_mean': surr_mean if not np.isnan(surr_mean) else np.nan,
        'surrogate_std': surr_std if not np.isnan(surr_std) else np.nan,
        'surrogate_median': surr_median if not np.isnan(surr_median) else np.nan,
        'n_surrogates': len(surrogates) if surrogates else 0,
        'alpha_class': alpha_class,
        'confidence_informational': confidence,
        'status': status,
        'has_trend': has_trend,
        'kendall_tau': tau_val,
        'kendall_p': p_trend,
        'is_stationary_adf': is_stationary if is_stationary is not None else None,
        'adf_pvalue': adf_pvalue if adf_pvalue is not None else np.nan,
        'adf_statistic': adf_stat if adf_stat is not None else np.nan,
        'method': 'Peng et al. (1994) DFA + Kantelhardt et al. (2002) MF-DFA'
    }
    
    # ============================================================
    # 12. CREATE AND SAVE FIGURES
    # ============================================================
    
    # FIGURE 1: Validation Graphics
    try:
        fig1 = create_validation_figure(
            layer.name(), field_name, best_order, alpha, intercept, r2,
            log_s, log_f, qs, hq_vals, mf_width, tau_q, 
            alpha_spectrum, f_alpha, surrogates, alphas_stab,
            ci_mean, surr_mean, p_val_text, effect, ts
        )
        
        output_path_val = OUTPUT_DIR / f"DFA_Validation_{safe_name}.png"
        fig1.savefig(str(output_path_val), dpi=150, bbox_inches='tight', 
                    facecolor='white')
        plt.close(fig1)
        print(f"\nValidation graphics saved to: {output_path_val}")
    except (OSError, ValueError) as e:
        print(f"Error saving validation figure: {e}")
        output_path_val = None
    
    # FIGURE 2: Results Summary
    try:
        fig2 = create_summary_figure(
            layer.name(), field_name, len(ts), best_order,
            min_scale, max_scale, len(scales), alpha,
            ci_lower, ci_upper, ci_mean, stability, r2,
            alpha_class, mf_class, stat_class, fit_class,
            mf_width, p_val_text, effect, bootstrap_reliability,
            confidence, bootstrap_bias, surr_mean, surr_std,
            iid_mean, status
        )
        
        output_path_res = OUTPUT_DIR / f"DFA_Results_{safe_name}.png"
        fig2.savefig(str(output_path_res), dpi=150, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig2)
        print(f"Results summary saved to: {output_path_res}")
    except (OSError, ValueError) as e:
        print(f"Error saving results figure: {e}")
        output_path_res = None
    
    # ============================================================
    # 13. EXPORT DETAILED EXCEL REPORT
    # ============================================================
    export_detailed_excel_report(
        OUTPUT_DIR, safe_name, results_dict,
        log_s, log_f, qs, hq_vals, hq_r2_vals,
        tau_q, alpha_spectrum, f_alpha,
        surrogates, alphas_stab, ts
    )
    
    # ============================================================
    # 14. SHOW RESULTS DIALOG
    # ============================================================
    result_msg = (
        f"Layer: {layer.name()}\n"
        f"Field: {field_name}\n"
        f"Method: Peng et al. (1994) + Kantelhardt et al. (2002)\n\n"
        f"α = {alpha:.4f}\n"
        f"Stationary bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n"
        f"R² = {r2:.4f} [{fit_class}]\n\n"
        f"CLASSIFICATION:\n"
        f"  {alpha_class}\n"
        f"  {mf_class}\n"
        f"  {stat_class}\n\n"
        f"Δh = {mf_width:.4f}\n"
        f"p-value = {p_val_text}\n"
        f"z-score = {effect:.3f}\n"
        f"Bootstrap bias = {bootstrap_bias:.4f}\n\n"
        f"FINAL STATUS: {status}\n\n"
        f"Output saved to:\n"
        f"  {OUTPUT_DIR}\n"
        f"Files:\n"
        f"  1) DFA_Validation_{safe_name}.png\n"
        f"  2) DFA_Results_{safe_name}.png\n"
        f"  3) DFA_Report_{safe_name[:50]}.xlsx"
    )
    
    if not np.isnan(iid_mean):
        result_msg += (f"\n\nIID bootstrap mean = {iid_mean:.3f} "
                      f"(shown for comparison only)")
    
    QMessageBox.information(None, "DFA Analysis Complete", result_msg)


# ============================================================
# RUN ANALYSIS
# ============================================================

if __name__ == "__main__":
    run_dfa_analysis()
else:
    run_dfa_analysis()