"""
HRV Feature Extraction from BVP
================================
Extracts time-domain and frequency-domain HRV features from the raw BVP
channel (64 Hz in WESAD, downsampled to 32 Hz after preprocessing).

Features extracted per window (12 total):
  Time domain (6):  mean_rr, sdnn, rmssd, pnn50, mean_hr, cv_rr
  Frequency domain (6): lf_power, hf_power, lf_hf_ratio, vlf_power,
                        total_power, hf_norm

These are the gold-standard biomarkers for autonomic nervous system stress
response validated across hundreds of studies.

Usage:
    from src.preprocessing.hrv_features import extract_hrv_features
    hrv = extract_hrv_features(bvp_window, fs=32)   # shape [12]
"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, welch


N_HRV_FEATURES = 12


def extract_hrv_features(bvp: np.ndarray, fs: int = 32) -> np.ndarray:
    """
    Extract 12 HRV features from a BVP window.

    Args:
        bvp:  1-D BVP signal (one channel of a windowed segment)
        fs:   sampling frequency in Hz

    Returns:
        features: np.ndarray of shape [12], float32
                  Returns zeros if peak detection fails (< 3 peaks).
    """
    features = np.zeros(N_HRV_FEATURES, dtype=np.float32)

    try:
        # ── R-peak detection ─────────────────────────────────────────────────
        # Use prominence-based peak detection on the normalised BVP
        bvp_norm = (bvp - bvp.mean()) / (bvp.std() + 1e-8)
        min_dist  = int(0.4 * fs)   # minimum 400 ms between beats (~150 bpm max)
        peaks, _  = find_peaks(bvp_norm, distance=min_dist, prominence=0.3)

        if len(peaks) < 3:
            return features   # not enough peaks — return zeros

        # ── RR intervals (milliseconds) ──────────────────────────────────────
        rr_ms = np.diff(peaks) / fs * 1000.0   # convert samples → ms

        # Filter physiologically plausible RR intervals (300–2000 ms = 30–200 bpm)
        rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
        if len(rr_ms) < 2:
            return features

        # ── Time-domain features ─────────────────────────────────────────────
        mean_rr   = float(np.mean(rr_ms))
        sdnn      = float(np.std(rr_ms))
        diff_rr   = np.diff(rr_ms)
        rmssd     = float(np.sqrt(np.mean(diff_rr ** 2)))
        pnn50     = float(np.mean(np.abs(diff_rr) > 50) * 100)   # % > 50ms
        mean_hr   = float(60000.0 / mean_rr) if mean_rr > 0 else 0.0
        cv_rr     = float(sdnn / mean_rr) if mean_rr > 0 else 0.0

        features[0] = mean_rr
        features[1] = sdnn
        features[2] = rmssd
        features[3] = pnn50
        features[4] = mean_hr
        features[5] = cv_rr

        # ── Frequency-domain features (Welch PSD) ────────────────────────────
        # Interpolate RR series to 4 Hz for spectral analysis
        if len(rr_ms) >= 4:
            rr_times   = np.cumsum(rr_ms) / 1000.0   # seconds
            interp_fs  = 4.0
            t_interp   = np.arange(rr_times[0], rr_times[-1], 1.0 / interp_fs)

            if len(t_interp) >= 8:
                rr_interp = np.interp(t_interp, rr_times, rr_ms)

                freqs, psd = welch(rr_interp, fs=interp_fs,
                                   nperseg=min(len(rr_interp), 64))

                def band_power(f_low, f_high):
                    mask = (freqs >= f_low) & (freqs < f_high)
                    return float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0

                vlf   = band_power(0.003, 0.04)
                lf    = band_power(0.04,  0.15)
                hf    = band_power(0.15,  0.40)
                total = vlf + lf + hf + 1e-8

                features[6]  = lf
                features[7]  = hf
                features[8]  = lf / (hf + 1e-8)   # LF/HF ratio
                features[9]  = vlf
                features[10] = total
                features[11] = hf / total           # HF normalised

    except Exception:
        # Silently return zeros — never crash the training loop
        pass

    return features


def extract_hrv_batch(X: np.ndarray, bvp_channel: int = 1, fs: int = 32) -> np.ndarray:
    """
    Extract HRV features for a batch of windows.

    Args:
        X:           shape [N, C, T]  — windowed sensor data
        bvp_channel: index of the BVP channel (default 1)
        fs:          sampling frequency

    Returns:
        hrv_features: shape [N, 12]
    """
    from tqdm import tqdm
    N = X.shape[0]
    out = np.zeros((N, N_HRV_FEATURES), dtype=np.float32)
    for i in tqdm(range(N), desc="HRV extraction", leave=False):
        out[i] = extract_hrv_features(X[i, bvp_channel], fs=fs)
    return out
