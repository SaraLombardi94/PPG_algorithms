# This Python file uses the following encoding: utf-8
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import sklearn
import matplotlib.pyplot as plt


class fiducialPointsDetection:
    """
    A class for detecting fiducial points in photoplethysmogram (PPG) signals 
    such as systolic peaks, waveform minima (valleys), and dicrotic notches using band-pass 
    filtering, peak detection, and signal processing techniques.

    Key Features:
    - Bandpass filtering to enhance peak detection.
    - Detection of systolic peaks using a sliding window and peak refinement.
    - Detection of valleys (diastolic minima) and dicrotic notches.
    - Filtering lag compensation for accurate peak location.

    Methods:
    ----------
    - butter_bandpass: Apply a band-pass filter to enhance signal.
    - find_systolic_peaks: Detect systolic peaks within a signal window.
    - sysDet: Detect all systolic peaks over the full signal using sliding windows.
    - refine_peaks_detection: Filter out spurious peaks based on amplitude thresholds.
    - remove_filter_lag: Compensate for lag introduced by filtering.
    - sysDet_check: Remove invalid systolic peaks based on SP_MAX time criteria.
    - valDet: Detect valleys (minima) between and around systolic peaks.
    - dicNotchDetect: Detect the dicrotic notch using derivative analysis.
    """
    
    cutoff_high = 3
    cutoff_low = 0.5
    maxBPM = 100
    window_duration = 5 # sec

    def __init__(self):
        pass

    @classmethod
    # band pass filtering to enhance systolic peaks and remove DC component
    def butter_bandpass(self,data,fs, cutoff_low, cutoff_high, order):
        """
        Apply a Butterworth bandpass filter to enhance the signal for peak detection.
        """
        nyq = 0.5 * fs
        normal_cutoff_low = cutoff_low / nyq
        normal_cutoff_high = cutoff_high / nyq
        b, a = signal.butter(order, [normal_cutoff_low, normal_cutoff_high], btype='bandpass', analog=False)
        filtered = signal.filtfilt(b, a, data)
        return filtered

    @classmethod
    # function to detect systolic peaks
    def find_systolic_peaks(self, window_flux, fs, cutoff_low, cutoff_high, order):
        """
        Detect systolic peaks within a window using filtering and find_peaks.
        Peaks are further refined to remove low-confidence detections.
        """
        # distance criteria
        thr = 60 / self.maxBPM
        filtered_flux = self.butter_bandpass(window_flux, fs, cutoff_low, cutoff_high, order)
        normalized_flux = sklearn.preprocessing.minmax_scale(filtered_flux, (-1, 1))
        loc_peak, ampl = find_peaks(normalized_flux, distance=int(thr * fs), height=0) #changed from 0.1 to 0
        if len(loc_peak) == 0:
            return [],[]

        loc_peak, rejected_peaks = self.refine_peaks_detection(normalized_flux, loc_peak)
        # Debug
        # plt.plot(normalized_flux)
        # plt.scatter(loc_peak,normalized_flux[loc_peak],color='green')
        # if len(rejected_peaks) !=0:
        #     plt.scatter(rejected_peaks,normalized_flux[rejected_peaks],color='red',marker="x")
        # plt.show()
        return loc_peak, rejected_peaks

    @classmethod
    def sysDet(self, flux, time, fs,cutoff_low, cutoff_high,order):
        """
        Detect systolic peaks across the full signal using sliding windows and filter lag correction.
        """
        overlap = 0.5
        window_size = int(self.window_duration*fs)
        step_size = int(window_size * (1 - overlap))
        if len(flux) > window_size:
            # find peaks using sliding windows
            locs = []
            for start in range(0, len(flux) - window_size, step_size):
                end = start + window_size
                window_flux = flux[start:end]
                loc_peak,rejected_peaks = self.find_systolic_peaks(window_flux,fs,cutoff_low, cutoff_high, order)
                if len(loc_peak) == 0:
                    continue
                locs.extend(loc_peak + start)

            last_window = flux[len(flux)-window_size:]
            loc_peak_last, rejected_peaks_last = self.find_systolic_peaks(last_window, fs, cutoff_low, cutoff_high, order)
            locs.extend(loc_peak_last + (len(flux)-window_size))
            locs = np.unique(np.array(locs))
        else:

            locs, rejected_peaks = self.find_systolic_peaks(flux, fs, cutoff_low, cutoff_high)


        locs = self.remove_filter_lag(fs,cutoff_high,cutoff_low,flux,locs)
        #TODO: controllare perchè ci sono doppioni (intorno troppo grande?)
        locs = np.unique(np.array(locs))

        sys = np.zeros((len(locs), 3))
        sys[:,0] = locs
        sys[:,1] = time[locs]
        sys[:,2] = flux[locs]
        return sys


    @classmethod
    def refine_peaks_detection(self,signal, peak_locs):
        """
        Refines detected peaks by comparing each peak to neighbors.
        Only keeps peaks that differ enough from neighbors to be considered valid.
        """
        candidate_peaks = []
        rejected_peaks = []
        if len(peak_locs)<=1:
            candidate_peaks.append(peak_locs[0])
            return np.array(candidate_peaks), np.array(rejected_peaks)
        # first peak is compared only to next value
        first_peak = signal[peak_locs[0]]
        second_peak = signal[peak_locs[1]]
        delta = abs(first_peak-second_peak)
        if delta >= max(first_peak, second_peak) * 0.45:
            if max(first_peak,second_peak) ==  first_peak:
                if peak_locs[0] not in candidate_peaks and peak_locs[0] not in rejected_peaks:
                    candidate_peaks.append(peak_locs[0])
            else:
                if peak_locs[0] not in rejected_peaks and peak_locs[0] not in candidate_peaks:
                    rejected_peaks.append(peak_locs[0])
        else:
            candidate_peaks.append(peak_locs[0])
        # intermediate values are compared with previous and next peak
        for i in range(1, len(peak_locs) -1):
            valore_corrente = signal[peak_locs][i]
            valore_precedente = signal[peak_locs][i - 1]
            valore_successivo = signal[peak_locs][i + 1]
            delta_prev = abs(valore_corrente - valore_precedente)
            delta_next = abs(valore_corrente - valore_successivo)
            threshold_pr = max(valore_corrente, valore_precedente) * 0.45
            threshold_fw = max(valore_corrente, valore_successivo) * 0.45
            if delta_prev >= threshold_pr and delta_next >= threshold_fw:
                if valore_corrente >= max(valore_precedente, valore_successivo):
                    if peak_locs[i] not in candidate_peaks and peak_locs[i] not in rejected_peaks:
                        candidate_peaks.append(peak_locs[i])
                    # if peak_locs[i - 1] not in rejected_peaks and peak_locs[i - 1] not in candidate_peaks:
                    #     rejected_peaks.append(peak_locs[i - 1])
                    # if peak_locs[i + 1] not in rejected_peaks and peak_locs[i + 1] not in candidate_peaks:
                    #     rejected_peaks.append(peak_locs[i + 1])
                else:
                    # if peak_locs[i - 1] not in candidate_peaks and peak_locs[i - 1] not in rejected_peaks:
                    #     candidate_peaks.append(peak_locs[i - 1])
                    if peak_locs[i] not in rejected_peaks and peak_locs[i] not in candidate_peaks:
                        rejected_peaks.append(peak_locs[i])
                    # if peak_locs[i + 1] not in rejected_peaks and peak_locs[i + 1] not in candidate_peaks:
                    #     rejected_peaks.append(peak_locs[i + 1])
            else:
                # if peak_locs[i - 1] not in candidate_peaks and peak_locs[i - 1] not in rejected_peaks:
                #     candidate_peaks.append(peak_locs[i - 1])
                if peak_locs[i] not in candidate_peaks and peak_locs[i] not in rejected_peaks:
                    candidate_peaks.append(peak_locs[i])
                # if peak_locs[i + 1] not in candidate_peaks and peak_locs[i + 1] not in rejected_peaks:
                #     candidate_peaks.append(peak_locs[i + 1])

        # first peak is compared only to next value
        last_peak = signal[peak_locs[-1]]
        previous_peak = signal[peak_locs[-2]]
        delta = abs(last_peak-previous_peak)
        if delta >= max(last_peak, previous_peak) * 0.45:
            if max(last_peak, previous_peak) == last_peak:
                if peak_locs[-1] not in candidate_peaks and peak_locs[-1] not in rejected_peaks:
                    candidate_peaks.append(peak_locs[-1])
                # if peak_locs[-2] not in rejected_peaks and peak_locs[-2] not in candidate_peaks:
                #     rejected_peaks.append(peak_locs[-2])
            else:
                # if peak_locs[-2] not in candidate_peaks and peak_locs[-2] not in rejected_peaks:
                #     candidate_peaks.append(peak_locs[-2])
                if peak_locs[-1] not in rejected_peaks and peak_locs[-1] not in candidate_peaks:
                    rejected_peaks.append(peak_locs[-1])
        else:
            candidate_peaks.append(peak_locs[-1])

        return np.array(candidate_peaks), np.array(rejected_peaks)



    @classmethod
    def remove_filter_lag(self, fs, cutoff_high, cutoff_low, flux, peak_locs):
        """
        Compensate for time lag introduced by the filter.
        For each peak, search nearby region for true maximum.
        """
        locs_flux = peak_locs

        epsilon = int(fs*(1/(2*(cutoff_high - cutoff_low)))/2)
        for idx in range(0,len(peak_locs)):
            if peak_locs[idx]< epsilon:
                intorno = flux[(peak_locs[idx] - int(epsilon/8)) : (peak_locs[idx] + int(epsilon/8) +1)]
                max_intorno = np.max(intorno)
                max_index = np.where(intorno==max_intorno)[0][0] + (peak_locs[idx] - int(epsilon/8))
            else:
                intorno = flux[(peak_locs[idx] - epsilon) : (peak_locs[idx] + epsilon+1)]
                max_intorno = np.max(intorno)
                max_index = np.where(intorno==max_intorno)[0][0] + (peak_locs[idx] - epsilon)
            locs_flux[idx] = max_index
        return np.array(locs_flux)

    # checks systolic time between wave onset and systolic peak
    # and selects only beats according to SP_MAX value criteria
    def sysDet_check(sys, val):
        """
        Remove systolic peaks where the delay from valley to peak exceeds the SP_MAX threshold (0.49).
        """
        N = sys.shape[0] # number of detected peaks
        delta = sys[:,0] - val[:-1,0]
        window = np.ones(N, dtype=bool)
        frac = 0.49 #SP_MAX value find in literature
        for i in range(N):
            if delta[i] > frac:
                window[i] = False
        sys = sys[window]
        val_ok = val[:-1][window]
        # riaggiungo l'ultimo elemento dell'array val sennò esclude l'ultima finestra
        val = np.vstack((val_ok, val[-1]))

        return sys, val


    def valDet(flux, time, sys):
        """
        Detect valleys (minima) between, before, and after systolic peaks.
        Returns corresponding values and time indices.
        """
        N = sys.shape[0]  # Numero di picchi sistolici rilevati
        val = np.zeros((N+1, 3))  # Ora gestisce un minimo extra prima e dopo
    
        # Trova il minimo prima del primo picco sistolico
        pre_win = time < sys[0, 1]
        if np.any(pre_win):
            flux_pre = flux[pre_win]
            time_pre = time[pre_win]
            val[0,0] = int(np.argmin(flux_pre))
            val[0, 2] = np.min(flux_pre)
            minPoint_pre = time_pre[flux_pre == val[0, 2]]
            val[0, 1] = minPoint_pre[0]
        else:
            val = val[1:]  # Se non c'è segnale prima, rimuoviamo questa riga
    
        # Trova i minimi tra i picchi sistolici
        for i in range(N-1):
            win = (time < sys[i+1, 1]) & (time >= sys[i, 1])
            start_index = sys[i,0]
            timeCycle = time[win]
            fluxCycle = flux[win]
            if len(timeCycle) == 0:
                continue
            val[i+1, 2] = np.min(fluxCycle)
            minPoint = timeCycle[fluxCycle == val[i+1, 2]]
            val[i+1, 1] = minPoint[0]
            val[i+1, 0] = int(np.argmin(fluxCycle))+start_index
    
        # Trova il minimo dopo l'ultimo picco sistolico
        post_win = time > sys[-1, 1]
        last_index = sys[-1,0]
        if np.any(post_win):
            flux_post = flux[post_win]
            time_post = time[post_win]
            val[-1, 2] = np.min(flux_post)
            minPoint_post = time_post[flux_post == val[-1, 2]]
            val[-1, 1] = minPoint_post[0]
            val[-1, 0] = int(np.argmin(flux_post))+ last_index
        else:
            val = val[:-1]  # Se non c'è segnale dopo, rimuoviamo questa riga
    
        if len(val) == 0:
            return None, None, None, None
    
        # Mantieni il segnale tra il primo e l'ultimo minimo trovato
        flux = flux[(time >= val[0, 1]) & (time <= val[-1, 1])]
        time = time[(time >= val[0, 1]) & (time <= val[-1, 1])]
    
        return val


    @classmethod
    def dicNotchDetect(self,flux, time, sys, fs, plot=False):
        #flux_smoothed = self.butter_bandpass(flux, fs, cutoff_low=0.05, cutoff_high=5, order=1)
        # 1st DERIVATIVE
        # 3-point differentiator
        dfI = np.gradient(flux, time)
        dfI = np.convolve(dfI, np.ones(3)/3, mode='same')
        dfI_min_window = dfI[:int(len(flux)/2)]
        abs_minimum_dfI = np.where(dfI_min_window==np.min(dfI_min_window))[0][0]
        time_minimum_dfI = time[abs_minimum_dfI]
        max_dfI = np.argmax(dfI[abs_minimum_dfI:int(len(flux)*0.8)]) + abs_minimum_dfI
        time_maximum_dfI = time[max_dfI]
        # 2nd DERIVATIVE
        dfII = np.gradient(dfI, time)
        dfII = np.convolve(dfII, np.ones(3)/3, mode='same')
        sys_index = np.argmax(flux)
        dic_win = dfII[abs_minimum_dfI:max_dfI]
        #if np.max(dic_win) > dic_win[0]:
        absmaxIndex = np.argmax(dic_win)
        maxIndex = absmaxIndex + abs_minimum_dfI
        dicnotch_index = maxIndex
        dicNotch_dI = time[dicnotch_index]
        peaks,_ = find_peaks(dic_win,height=(dfII[maxIndex]*0.5))
        # check whether peaks next to the abs maximum are best candidate for dicNotch (lower flux value)
        if len(peaks)>0:
            dicValleyIndexes = peaks + abs_minimum_dfI
            dicValleyValues = flux[dicValleyIndexes]
            flux_min_index = np.argmin(dicValleyValues)
            best_candidate = dicValleyIndexes[flux_min_index]
            flux_min_value = flux[best_candidate]
            if flux_min_value < flux[dicnotch_index]:
                dicnotch_index = best_candidate 
                dicNotch_dI = time[best_candidate ]
        else:
            # f' 0 crossings (- +) > Local minima (after systole, below 80 % of peak flux)
            zero_crossings = np.where(np.diff(np.signbit(dfI)))[0]
            #zero_dfI = time[np.where(np.diff(sgn) > 0)[0]]
            zero_dfI = time[zero_crossings]
            #devo selezionare quelli >= al picco sistolico
            zero_dfI = zero_dfI[(zero_dfI >= sys) & (flux[zero_crossings] < np.max(flux) * 0.9) & (zero_dfI<time[int(len(time)*0.8)])]
            dicNotch_dI = np.min(zero_dfI)
            candidate_index = int(np.where(time==dicNotch_dI)[0])
            if flux[candidate_index]< flux[dicnotch_index]:
                dicnotch_index = candidate_index
                dicNotch_dI = time[candidate_index]
        if plot:
            fig, axs = plt.subplots(3, figsize=(6, 8))
            axs[0].plot(time,flux, color="blue")
            axs[1].plot(time,dfI,color="orange")
            axs[2].plot(time,dfII,color="purple")
            axs[1].scatter(time_minimum_dfI,dfI[abs_minimum_dfI],color="black")
            axs[1].scatter(time_maximum_dfI,dfI[max_dfI],color="red")
            axs[0].scatter(dicNotch_dI,flux[dicnotch_index],color="green")
            axs[1].scatter(dicNotch_dI,dfI[dicnotch_index],color="green")
            axs[2].scatter(dicNotch_dI,dfII[dicnotch_index],color="green")
            plt.show()

        return dicnotch_index, dicNotch_dI



    @classmethod
    def dicNotchDetect_old(self,flux, time, sys, fs):
        #flux_smoothed = self.butter_bandpass(flux, fs, cutoff_low=0.05, cutoff_high=5, order=1)
        # 1st DERIVATIVE
        # 3-point differentiator
        dfI = np.gradient(flux, time)
        #dfI = self.butter_bandpass(dfI,fs, cutoff_low=0.05, cutoff_high=5, order=1)
        dfI = np.convolve(dfI, np.ones(3)/3, mode='same') #convolution with 3x3 kernel, using "same" padding to have same number of samples
        #num_lost_samples = len(flux) - len(dfI)
        #dfI = np.concatenate((dfI[0]*np.ones(int(num_lost_samples/2)), dfI, dfI[-1]*np.ones(int(num_lost_samples/2))))  
        dfI_min_window = dfI[:int(len(flux)/2)]
        abs_minimum = np.where(dfI_min_window==np.min(dfI_min_window))[0]
        time_minimum = time[abs_minimum]
        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        axs[0].plot(time,flux, color="blue")
        axs[1].plot(time,dfI,color="orange")
        
        # f' 0 crossings (- +) > Local minima (after systole, below 80 % of peak flux)
        zero_crossings = np.where(np.diff(np.signbit(dfI)))[0]
        #zero_dfI = time[np.where(np.diff(sgn) > 0)[0]]
        zero_dfI = time[zero_crossings]
        #devo selezionare quelli >= al picco sistolico
        zero_dfI = zero_dfI[(zero_dfI >= sys) & (flux[zero_crossings] < np.max(flux) * 0.9) & (zero_dfI<time[int(len(time)*0.8)])]

        # f' local (negative) maxima
        maxLocs = []

        if len(zero_dfI) > 0:  # se c'è uno zero crossing ricerca alla sua sx
            maxWin = (time > sys) & (time < zero_dfI[0]) & (time>time_minimum)
            zero_dfI_index = np.where(np.isin(time, zero_dfI))[0]
            if not np.any(maxWin):
                maxWin = (time > sys) & (time < zero_dfI[0])
                zero_dfI_index = np.where(np.isin(time, zero_dfI))[0]
                print("ATTENZIONE: la finestra maxWin è vuota.")
                print(f"time range: {time.min()} - {time.max()}")
                print(f"sys: {sys}, zero_dfI[0]: {zero_dfI[0]}, time_minimum: {time_minimum}")
            dfII = np.gradient(dfI[maxWin],time[maxWin])
            maxIndex = np.where(dfII==np.max(dfII))[0]
            #axs[1].scatter(time[maxWin][maxIndex],dfI[maxWin][maxIndex],color="red")
            axs[1].scatter(zero_dfI, dfI[zero_dfI_index],color="black")
        else:  # altrimenti ricerca a dx del picco sistolico, fino alla fine del ciclo
            maxWin = (time > sys) & (time < time[int(len(time)*0.8)]) & (time>time_minimum)
            maxLocs, maxVals = find_peaks(dfI[maxWin])
            #maxLocs = maxLocs[(flux[np.isin(time, time[maxWin][maxLocs])]< np.max(flux) * 0.9)]
            win = (time < time[maxWin][maxLocs[0]]) & (time>time_minimum)
            dfII = np.gradient(dfI[win],time[win])
            maxIndex = np.where(dfII==np.max(dfII))[0]
            axs[1].scatter(time[win][maxIndex],dfI[win][maxIndex],color="red")



        if len(zero_dfI) == 0 and len(maxLocs) == 0:
            dicNotch_dI = time[int(np.floor(len(time)-len(time)/4))]
        elif len(zero_dfI) == 0:
            dicNotch_dI = time[win][maxIndex][0]
        elif len(maxLocs) == 0:
            dicNotch_dI = np.min(zero_dfI)
        else:
            dicNotch_dI = np.min([zero_dfI.min(), maxLocs.min()])
        dicnotch_index = int(np.where(time==dicNotch_dI)[0])
        axs[0].scatter(dicNotch_dI,flux[dicnotch_index])
        plt.show()

        return dicnotch_index, dicNotch_dI


