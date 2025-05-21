# This Python file uses the following encoding: utf-8
import numpy as np
import scipy
import pandas as pd
from fiducialPointsDetector import fiducialPointsDetector
from plotModelFitting import plotModelFitting
from PulseFitter import PulseFitter
import matplotlib.pyplot as plt

from scipy.interpolate import splrep, splev

class DataProcessor:
    def __init__(self, filepath, fs, fitting_type="Exp",
                 use_filter=False, cutoff_low=None, cutoff_high=None, filter_order=None):
        self.filepath = filepath
        self.fs = fs
        self.fitting_type = fitting_type
        self.use_filter = use_filter
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high
        self.filter_order = filter_order

        # Outputs
        self.signal = None
        self.time = None
        self.trend = None
        self.parameters = {}
        self.R2 = None
        self.NRMSE = None
        self.MSE = None
        self.figure = None
 
    def process_signal(self, plot=True):
        self.time, self.signal = self.load_signal()
        """Use external class to detect sys and val points."""
        detector = fiducialPointsDetector(
            self.signal,
            self.time,
            self.fs,
            df_cutoff_low=0.5,
            df_cutoff_high=3,
            df_order=2,
            minBPM=40,
            maxBPM=120
        )
        self.sys_peaks = detector.detect_systolic_peaks()
        self.valleys = detector.detect_valleys(self.sys_peaks)
        print(f"valleys: {self.valleys[:,0]}")
        #self.signal =  self.signal[(self.time >= self.valleys[0, 1]) & (self.time <= self.valleys[-1, 1])]
        #self.time = self.time[(self.time >= self.valleys[0, 1]) & (self.time <=  self.valleys[-1, 1])]
        # need at least 3 valleys to interpolate the signal to calculate trend
        if self.sys_peaks is None or self.valleys is None or len(self.sys_peaks) < 1 or len(self.valleys) < 4:
            return None, None, None, None, None
        # interpolate trend of PPG signal
        tck = splrep(self.valleys[:, 1], self.valleys[:, 2], k=3)
        self.trend = splev(self.time, tck)

        N = self.valleys.shape[0]
        model = np.zeros_like(self.signal)
        waves = np.zeros((len(self.signal), 3))
        R2 = np.zeros(N - 1)
        NRMSE = np.zeros(N - 1)
        MSE = np.zeros(N - 1)
        dicNotch = np.zeros((N - 1, 2))
        parameters = {}

        # pulse segmentation
        for i in range(N - 1):
            # mask for pulse segmentation
            mask = (self.time >= self.valleys[i, 1]) & (self.time <= self.valleys[i + 1, 1])
            time_segment = self.time[mask]
            flux_segment = self.signal[mask]
            trend_segment = self.trend[mask]

            if len(time_segment) < round((60/detector.maxBPM)*self.fs) or len(time_segment) > round((60/detector.minBPM)*self.fs):
                continue  # Skip pulse

            t0 = time_segment[0]
            time_relative = time_segment - t0
            syst = [self.sys_peaks[i, 1] - t0, self.sys_peaks[i, 2]]
            vall = [self.valleys[i, 1] - t0, self.valleys[i, 2]]
            # find dicrotic notch within the detrended pulse
            flux_segment_dt = flux_segment - trend_segment
            dn_idx, dn_time = detector.find_dicrotic_notch(flux_segment_dt,time_relative,syst[0])
            # amplitude on the original pulse
            dn_ampl = flux_segment[dn_idx]

            if dn_idx is None:
                continue # Skip pulse
            dicNotch[i] = [dn_time+t0, dn_ampl]
            # modelling of each pulse
            model_i, waves_i, parameters_i, r2_i = PulseFitter.fit_cycle(flux_segment,time_relative,trend_segment,dn_time,self.fitting_type)
            if model_i is None:
                R2[i] = None
                NRMSE[i] = None
                MSE[i] = None
            else:
                R2[i] = r2_i
                NRMSE[i] = parameters_i['nrmse']
                MSE[i] = parameters_i['mse']
                model[mask] = model_i
                waves[mask,:] = waves_i[:,:]
                parameters_i['pulse_index']=i
                parameters_i['start_time'] = t0
                parameters_i['end_time'] = time_segment[-1]
                self.parameters = self.update_parameters(self.parameters, parameters_i)

        self.R2, self.NRMSE, self.MSE = R2, NRMSE, MSE
        # check if the R2 of fitting is good
        idx_misfit = np.zeros(R2.shape[0], dtype=bool)
        bad_fit_location = np.where((np.array(R2) < 0.8) | (R2 is None))
        meanR2, meanNRMSE, meanMSE = self.fitEval(R2, NRMSE, MSE, idx_misfit)
        title = "Multi-exp model" if self.fitting_type =="Exp" else "Skew gaussian model"
        fig = plotModelFitting.plot_signal_with_fiducials_interactive(
            signal=self.signal,
            time=self.time,
            sys=self.sys_peaks,
            val=self.valleys,
            dic_notches = dicNotch,
            trend = self.trend,
            waves = waves,
            model = model,
            idx_misfit = idx_misfit, 
            R2 = R2,
            title = title
        )
        
        return self.parameters, meanR2, meanNRMSE, meanMSE, fig

    def load_signal(self):
        suffix = self.filepath.split('.')[-1].lower()
        if suffix == 'npy':
            data = np.load(self.filepath)
        elif suffix == 'npz':
            npz = np.load(self.filepath)
            data = npz['arr_0'] if 'arr_0' in npz else npz['data']
        elif suffix == 'csv':
            df = pd.read_csv(self.filepath)
            if 'PPG' in df.columns:
                data = df['PPG'].values
            elif 'pleth' in df.columns:
                data = df['pleth'].values
            else:
                raise ValueError("CSV file must contain 'PPG' or 'pleth' column.")
        else:
            raise ValueError("Unsupported file format.")

        if self.use_filter:
            print(f"before: {data[0:10]}")
            data = self._apply_filter(data)

        self.signal = data
        print(f"after: {self.signal[0:10]}")
        self.time = np.arange(len(self.signal)) / self.fs
        return self.time, self.signal

    def _apply_filter(self, data):
        print(f"fs is {self.fs},cutoff_low is {self.cutoff_low}, cutoff high is {self.cutoff_high}, order is {self.filter_order}")
        nyq = 0.5 * self.fs
        b, a = scipy.signal.butter(
            self.filter_order,
            [self.cutoff_low / nyq, self.cutoff_high / nyq],
            btype='bandpass',analog=False
        )
        return scipy.signal.filtfilt(b, a, data)

    def update_parameters(self, Par, cyclepar):
        for key, value in cyclepar.items():
            if key not in Par.keys():
                Par[key]= []
            Par[key].append(value)
        return Par

    def fitEval(self, R2, NRMSE, MSE, idx_misfit):
        R2 = R2[idx_misfit==False]
        NRMSE = NRMSE[idx_misfit==False]
        MSE = MSE[idx_misfit==False]
        meanR2 = np.mean(R2)
        meanNRMSE = np.mean(NRMSE)
        meanMSE = np.mean(MSE)
        return meanR2, meanNRMSE, meanMSE
