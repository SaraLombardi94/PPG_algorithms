

# This Python file uses the following encoding: utf-8
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import sklearn
import matplotlib.pyplot as plt

d_cutoff_low = 0.5
d_cutoff_high = 3
d_order = 1
max_bpm = 140

class FiducialPointsDetector:
    def __init__(self):
        pass
    
    @classmethod
    def detect_systolic_peaks(self, signal, time, fs):
        overlap = 0.5
        window_size = int(5 * fs)
        step_size = int(window_size * (1 - overlap))
        locs = []

        if len(signal) > window_size:
            for start in range(0, len(signal) - window_size, step_size):
                window = signal[start:start + window_size]
                peaks, _ = self._find_systolic_in_window(window,fs)
                locs.extend(peaks + start)

            window = signal[-window_size:]
            peaks, _ = self._find_systolic_in_window(window,fs)
            locs.extend(peaks + len(signal) - window_size)
        else:
            peaks, _ = self._find_systolic_in_window(window,fs)
            locs = peaks

        locs = self._remove_filter_lag(signal, fs, np.array(locs))
        locs = np.unique(locs)
        sys = np.zeros((len(locs), 3))
        sys[:, 0] = locs
        sys[:, 1] = time[locs]
        sys[:, 2] = signal[locs]
        return sys
    
    @classmethod
    def _butter_bandpass(self, data, fs):
        nyq = 0.5 * fs
        b, a = signal.butter(d_order, [d_cutoff_low / nyq, d_cutoff_high / nyq], btype='bandpass')
        return signal.filtfilt(b, a, data)
    
    @classmethod
    def _find_systolic_in_window(self, signal_window, fs):
        filtered = self._butter_bandpass(signal_window, fs)
        norm = sklearn.preprocessing.minmax_scale(filtered, (-1, 1))
        distance = int(fs * 60 / max_bpm)
        peaks, _ = signal.find_peaks(norm, distance=distance, height=0)
        candidate_peaks, rejected_peaks = self._refine_peaks(norm, peaks)
        return candidate_peaks, rejected_peaks
    
    @classmethod
    def _refine_peaks(self, signal, peak_locs):
        candidate_peaks = []
        rejected_peaks = []

        if len(peak_locs) <= 1:
            candidate_peaks.append(peak_locs[0])
            return np.array(candidate_peaks), np.array(rejected_peaks)

        first_val = signal[peak_locs[0]]
        second_val = signal[peak_locs[1]]
        delta = abs(first_val - second_val)
        if delta >= max(first_val, second_val) * 0.45:
            if first_val >= second_val:
                candidate_peaks.append(peak_locs[0])
            else:
                rejected_peaks.append(peak_locs[0])
        else:
            candidate_peaks.append(peak_locs[0])

        for i in range(1, len(peak_locs) - 1):
            current_val = signal[peak_locs[i]]
            prev_val = signal[peak_locs[i - 1]]
            next_val = signal[peak_locs[i + 1]]

            delta_prev = abs(current_val - prev_val)
            delta_next = abs(current_val - next_val)
            threshold_prev = max(current_val, prev_val) * 0.45
            threshold_next = max(current_val, next_val) * 0.45

            if delta_prev >= threshold_prev and delta_next >= threshold_next:
                if current_val >= max(prev_val, next_val):
                    candidate_peaks.append(peak_locs[i])
                else:
                    rejected_peaks.append(peak_locs[i])
            else:
                candidate_peaks.append(peak_locs[i])

        last_val = signal[peak_locs[-1]]
        prev_val = signal[peak_locs[-2]]
        delta = abs(last_val - prev_val)
        if delta >= max(last_val, prev_val) * 0.45:
            if last_val >= prev_val:
                candidate_peaks.append(peak_locs[-1])
            else:
                rejected_peaks.append(peak_locs[-1])
        else:
            candidate_peaks.append(peak_locs[-1])

        return np.array(candidate_peaks), np.array(rejected_peaks)
    
    @classmethod
    def _remove_filter_lag(self, signal,fs, peak_locs):
        epsilon = int(fs * (1 / (2 * (d_cutoff_high - d_cutoff_low))) / 2)
        corrected_locs = []

        for idx in peak_locs:
            if idx < epsilon:
                local_win = signal[max(0, idx - epsilon // 8): idx + epsilon // 8 + 1]
                start_idx = max(0, idx - epsilon // 8)
            else:
                local_win = signal[idx - epsilon: idx + epsilon + 1]
                start_idx = idx - epsilon

            if len(local_win) == 0:
                corrected_locs.append(idx)
                continue

            local_max_idx = np.argmax(local_win)
            corrected_idx = start_idx + local_max_idx
            corrected_locs.append(corrected_idx)

        return np.unique(np.array(corrected_locs))
    
    @classmethod
    def detect_valleys(self, signal, time, sys_peaks):
        N = sys_peaks.shape[0]
        val = np.zeros((N + 1, 3))

        pre = time < sys_peaks[0, 1]
        if np.any(pre):
            flux_pre = signal[pre]
            time_pre = time[pre]
            idx = np.argmin(flux_pre)
            val[0] = [idx, time_pre[idx], flux_pre[idx]]
        else:
            val = val[1:]

        for i in range(N - 1):
            mask = (time >= sys_peaks[i, 1]) & (time < sys_peaks[i + 1, 1])
            if not np.any(mask):
                continue
            flux_seg = signal[mask]
            time_seg = time[mask]
            idx = np.argmin(flux_seg)
            start = np.where(time == time_seg[0])[0][0]
            val[i + 1] = [start + idx, time_seg[idx], flux_seg[idx]]

        post = time > sys_peaks[-1, 1]
        if np.any(post):
            flux_post = signal[post]
            time_post = time[post]
            idx = np.argmin(flux_post)
            start = np.where(time == time_post[0])[0][0]
            val[-1] = [start + idx, time_post[idx], flux_post[idx]]
        else:
            val = val[:-1]

        return val

    @classmethod
    def find_dicrotic_notch(self, flux, time, sys_peak_time):
        dfI = np.gradient(flux, time)
        #dfI = np.convolve(dfI, np.ones(3)/3, mode='same')
        sys_peak_index = np.where(time==sys_peak_time)[0][0]
        dfI_min_window = dfI[sys_peak_index:int(len(flux)*0.7)]
        if len(dfI_min_window) == 0:
            return None, None
        abs_min_idx = np.argmin(dfI_min_window) + sys_peak_index
        max_dfI = np.argmax(dfI[abs_min_idx:int(len(flux)*0.8)]) + abs_min_idx
        # search area of the first derivative
        dfI_win = dfI[abs_min_idx:max_dfI]
        if len(dfI_win) <10:
            return None, None
        # zero crossing where the slope changes from negative to positive
        zero_crossings = np.where((dfI_win[:-1] < 0) & (dfI_win[1:] >= 0))[0]
        if len(zero_crossings) > 0:
            best_candidate = zero_crossings[0] + abs_min_idx
            dicnotch_index = best_candidate
            dicnotch_time = time[dicnotch_index]
        else:
            # find max in 2nd derivative
            dfII = np.gradient(dfI, time)
            #dfII = np.convolve(dfII, np.ones(3)/3, mode='same')
            dic_win = dfII[abs_min_idx:max_dfI]
            abs_max_idx = np.argmax(dic_win)
            max_idx = abs_max_idx + abs_min_idx
            dicnotch_index = max_idx
            dicnotch_time = time[dicnotch_index]
        return dicnotch_index, dicnotch_time
