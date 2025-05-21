# This Python file uses the following encoding: utf-8
import numpy as np
from sklearn import preprocessing
import scipy
import matplotlib.pyplot as plt

class fiducialPointsDetector:
    def __init__(self, signal, time, fs, df_cutoff_low=0.5, df_cutoff_high=3.0, df_order=2, minBPM=40, maxBPM=140):
        self.signal = signal
        self.time = time
        self.fs = fs
        self.cutoff_low = df_cutoff_low
        self.cutoff_high = df_cutoff_high
        self.order = df_order
        self.minBPM = minBPM
        self.maxBPM = maxBPM


    def detect_systolic_peaks(self):
        """Detect systolic peaks using sliding windows and filtering."""
        overlap = 0.5
        window_size = int(5 * self.fs)
        step_size = int(window_size * (1 - overlap))
        locs = []

        if len(self.signal) > window_size:
            for start in range(0, len(self.signal) - window_size, step_size):
                window = self.signal[start:start + window_size]
                peaks, _ = self._find_systolic_in_window(window)
                locs.extend(peaks + start)

            # Handle last window
            window = self.signal[-window_size:]
            peaks, _ = self._find_systolic_in_window(window)
            locs.extend(peaks + len(self.signal) - window_size)
        else:
            peaks, _ = self._find_systolic_in_window(self.signal)
            locs = peaks
        locs_without_lag = self._remove_filter_lag(np.array(locs))
        locs_unique = np.unique(locs_without_lag)
        sys = np.zeros((len(locs_unique), 3))
        sys[:, 0] = locs_unique
        sys[:, 1] = self.time[locs_unique]
        sys[:, 2] = self.signal[locs_unique]
        return sys

    def _butter_bandpass(self, data):
        nyq = 0.5 * self.fs
        b, a = scipy.signal.butter(self.order, [self.cutoff_low / nyq, self.cutoff_high / nyq], btype='bandpass')
        return scipy.signal.filtfilt(b, a, data)

    def _find_systolic_in_window(self, signal_window):
        """Apply filtering and find peaks in a window."""
        filtered = self._butter_bandpass(signal_window)
        norm = preprocessing.minmax_scale(filtered, (-1, 1))
        distance = int(self.fs * 60 / self.maxBPM)
        peaks, props = scipy.signal.find_peaks(norm, distance=distance, height=0)
        candidate_peaks, rejected_peaks = self._refine_peaks(norm, peaks)
        return candidate_peaks, rejected_peaks

    def _refine_peaks(self, signal, peak_locs):
        """
        Refines detected peaks by comparing each peak to neighbors.
        Keeps only peaks that differ enough from neighbors to be considered valid.
        """
        candidate_peaks = []
        rejected_peaks = []

        if len(peak_locs) <= 1:
            candidate_peaks.append(peak_locs[0])
            return np.array(candidate_peaks), np.array(rejected_peaks)

        # First peak → compare only to next
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

        # Intermediate peaks → compare to prev and next
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

        # Last peak → compare only to previous
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


    def _remove_filter_lag(self, peak_locs):
        epsilon = int(self.fs * (1 / (2 * (self.cutoff_high - self.cutoff_low))) / 2)
        corrected_locs = []

        for idx in peak_locs:
            if idx < epsilon:
                local_win = self.signal[max(0, idx - epsilon // 8): idx + epsilon // 8 + 1]
                start_idx = max(0, idx - epsilon // 8)
            else:
                local_win = self.signal[idx - epsilon: idx + epsilon + 1]
                start_idx = idx - epsilon

            if len(local_win) == 0:
                corrected_locs.append(idx)
                continue

            local_max_idx = np.argmax(local_win)
            corrected_idx = start_idx + local_max_idx
            corrected_locs.append(corrected_idx)

        return np.unique(np.array(corrected_locs))

    def detect_valleys(self, sys_peaks):
        """
        Detect valleys (minima) between, before, and after systolic peaks.
        Returns corresponding values and time indices.
        """
        N = sys_peaks.shape[0]
        val = np.zeros((N + 1, 3))

        # Before first peak
        pre = self.time < sys_peaks[0, 1]
        if np.any(pre):
            flux_pre = self.signal[pre]
            time_pre = self.time[pre]
            idx = np.argmin(flux_pre)
            val[0] = [idx, time_pre[idx], flux_pre[idx]]
        else:
            val = val[1:]

        # Between peaks
        for i in range(N - 1):
            mask = (self.time >= sys_peaks[i, 1]) & (self.time < sys_peaks[i + 1, 1])
            if not np.any(mask): continue
            flux_seg = self.signal[mask]
            time_seg = self.time[mask]
            idx = np.argmin(flux_seg)
            start = np.where(self.time == time_seg[0])[0][0]
            val[i + 1] = [start + idx, time_seg[idx], flux_seg[idx]]

        # After last peak
        post = self.time > sys_peaks[-1, 1]
        if np.any(post):
            flux_post = self.signal[post]
            time_post = self.time[post]
            idx = np.argmin(flux_post)
            start = np.where(self.time == time_post[0])[0][0]
            val[-1] = [start + idx, time_post[idx], flux_post[idx]]
        else:
            val = val[:-1]

        return val

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

        #     peaks, _ = find_peaks(dic_win, height=(dfII[max_idx]*0.5))
        #     if len(peaks) > 0:
        #         dic_indices = peaks + abs_min_idx
        #         dic_values = flux[dic_indices]
        #         min_idx = np.argmin(dic_values)
        #         best_candidate = dic_indices[min_idx]
        #         if flux[best_candidate] < flux[dicnotch_index]:
        #             dicnotch_index = best_candidate
        #             dicnotch_time = time[best_candidate]
        # else:
        #     zero_crossings = np.where(np.diff(np.signbit(dfI)))[0]
        #     zero_times = time[zero_crossings]
        #     valid = (zero_times >= sys_peak_time) & (flux[zero_crossings] < np.max(flux) * 0.9) & (zero_times < time[int(len(time)*0.8)])
        #     zero_times = zero_times[valid]
        #     if len(zero_times) > 0:
        #         dicnotch_time = np.min(zero_times)
        #         candidate_idx = int(np.where(time == dicnotch_time)[0])
        #         if flux[candidate_idx] < flux[dicnotch_index]:
        #             dicnotch_index = candidate_idx
        #             dicnotch_time = time[candidate_idx]

        return dicnotch_index, dicnotch_time

#    def find_dicrotic_notch(self, flux, time, sys_peak_time):
#        dfI = np.gradient(flux, time)
#        dfI = np.convolve(dfI, np.ones(3)/3, mode='same')
#        dfI_min_window = dfI[:int(len(flux)/2)]
#        abs_min_idx = np.argmin(dfI_min_window)
#        max_dfI = np.argmax(dfI[abs_min_idx:int(len(flux)*0.8)]) + abs_min_idx

#        dfII = np.gradient(dfI, time)
#        dfII = np.convolve(dfII, np.ones(3)/3, mode='same')
#        dic_win = dfII[abs_min_idx:max_dfI]
#        if len(dic_win)<10:
#            print(dic_win)
#        abs_max_idx = np.argmax(dic_win)
#        max_idx = abs_max_idx + abs_min_idx
#        dicnotch_index = max_idx
#        dicnotch_time = time[dicnotch_index]

#        peaks, _ = scipy.signal.find_peaks(dic_win, height=(dfII[max_idx]*0.5))
#        if len(peaks) > 0:
#            dic_indices = peaks + abs_min_idx
#            dic_values = flux[dic_indices]
#            min_idx = np.argmin(dic_values)
#            best_candidate = dic_indices[min_idx]
#            if flux[best_candidate] < flux[dicnotch_index]:
#                dicnotch_index = best_candidate
#                dicnotch_time = time[best_candidate]
#        else:
#            zero_crossings = np.where(np.diff(np.signbit(dfI)))[0]
#            zero_times = time[zero_crossings]
#            valid = (zero_times >= sys_peak_time) & (flux[zero_crossings] < np.max(flux) * 0.9) & (zero_times < time[int(len(time)*0.8)])
#            zero_times = zero_times[valid]
#            if len(zero_times) > 0:
#                dicnotch_time = np.min(zero_times)
#                candidate_idx = int(np.where(time == dicnotch_time)[0])
#                if flux[candidate_idx] < flux[dicnotch_index]:
#                    dicnotch_index = candidate_idx
#                    dicnotch_time = time[candidate_idx]

#        return dicnotch_index, dicnotch_time

#    def find_dicrotic_notch(self,flux, time, sys, f):
#        #flux_smoothed = self.butter_bandpass(flux, fs, cutoff_low=0.05, cutoff_high=5, order=1)
#        # 1st DERIVATIVE
#        # 3-point differentiator
#        dfI = np.gradient(flux, time)
#        dfI = np.convolve(dfI, np.ones(3)/3, mode='same')
#        dfI_min_window = dfI[:int(len(flux)/2)]
#        abs_minimum_dfI = np.where(dfI_min_window==np.min(dfI_min_window))[0][0]
#        time_minimum_dfI = time[abs_minimum_dfI]
#        max_dfI = np.argmax(dfI[abs_minimum_dfI:int(len(flux)*0.8)]) + abs_minimum_dfI
#        time_maximum_dfI = time[max_dfI]
#        # 2nd DERIVATIVE
#        dfII = np.gradient(dfI, time)
#        dfII = np.convolve(dfII, np.ones(3)/3, mode='same')
#        sys_index = np.argmax(flux)
#        #print(f"{abs_minimum_dfI},{max_dfI}")
#        dic_win = dfII[abs_minimum_dfI:max_dfI]
#        #if np.max(dic_win) > dic_win[0]:
#        absmaxIndex = np.argmax(dic_win)
#        maxIndex = absmaxIndex + abs_minimum_dfI
#        dicnotch_index = maxIndex
#        dicNotch_dI = time[dicnotch_index]
#        peaks,_ = scipy.signal.find_peaks(dic_win,height=(dfII[maxIndex]*0.5))
#        # check whether peaks next to the abs maximum are best candidate for dicNotch (lower flux value)
#        if len(peaks)>0:
#            dicValleyIndexes = peaks + abs_minimum_dfI
#            dicValleyValues = flux[dicValleyIndexes]
#            flux_min_index = np.argmin(dicValleyValues)
#            best_candidate = dicValleyIndexes[flux_min_index]
#            flux_min_value = flux[best_candidate]
#            if flux_min_value < flux[dicnotch_index]:
#                dicnotch_index = best_candidate
#                dicNotch_dI = time[best_candidate ]
#        else:
#            # f' 0 crossings (- +) > Local minima (after systole, below 80 % of peak flux)
#            zero_crossings = np.where(np.diff(np.signbit(dfI)))[0]
#            #zero_dfI = time[np.where(np.diff(sgn) > 0)[0]]
#            zero_dfI = time[zero_crossings]
#            #devo selezionare quelli >= al picco sistolico
#            zero_dfI = zero_dfI[(zero_dfI >= sys) & (flux[zero_crossings] < np.max(flux) * 0.9) & (zero_dfI<time[int(len(time)*0.8)])]
#            dicNotch_dI = np.min(zero_dfI)
#            candidate_index = int(np.where(time==dicNotch_dI)[0])
#            if flux[candidate_index]< flux[dicnotch_index]:
#                dicnotch_index = candidate_index
#                dicNotch_dI = time[candidate_index]

#        return dicnotch_index, dicNotch_dI
