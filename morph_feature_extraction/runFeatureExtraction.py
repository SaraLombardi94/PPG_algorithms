# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:48:49 2025
@author: adhk198
"""

import pandas as pd
import numpy as np 
import os
from glob import glob
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import argrelmax, argrelmin, argrelextrema
import sklearn
from sklearn import preprocessing
from scipy.interpolate import splrep, splev
from FiducialPointsDetector import FiducialPointsDetector as FPD
from MorphologicalFeatureExtractor import MorphologicalFeatureExtractor as MFE
#from morphologicalFeatureExtractor import MorphologicalFeatureExtractor
from pathlib import Path



dataDir = r"D:\ArterialStiffnessProjectSara\dataset\in-vitro-data\dati"
mainDirFeatures = r"D:\ArterialStiffnessProjectSara\dataset\in-vitro-data\features"
dataPaths = glob(os.path.join(dataDir,"*PPG_IR*"))
SAMPLE_RATE = 2000 #Hz

# duration of systolic phase 
SP_MIN = 0.08*SAMPLE_RATE
SP_MAX = 0.49*SAMPLE_RATE
#physiological parameters
HR_MIN = 40 #bpm
HR_MAX = 140 #bpm
# min and max pulse duration 
PWD_MIN = round((60/HR_MAX)*SAMPLE_RATE) #samples
PWD_MAX = round((60/HR_MIN)*SAMPLE_RATE)

USE_FILTER = True
REMOVE_DC = False
NORMALISE = False
#filter parameters
CUT_OFF_LOW = 0.05
CUT_OFF_HIGH = 10
ORDER = 2
DEBUGGER_PLOTS = False

if USE_FILTER == False:
    outDir = os.path.join(mainDirFeatures,f"no_filter")
    os.makedirs(outDir,exist_ok=True)
else:
    outDir = os.path.join(mainDirFeatures,f"butter-{CUT_OFF_LOW}-{CUT_OFF_HIGH}_order{ORDER}")
    os.makedirs(outDir,exist_ok=True)

# def butter_bandpass(data, fs, cutoff_low=0.05, cutoff_high=10, order=2):
#     nyq = 0.5 * fs

#     if cutoff_low == 0:
#         # Low pass filter
#         normal_cutoff_high = cutoff_high / nyq
#         b, a = signal.butter(order, normal_cutoff_high, btype='low', analog=False)
#     else:
#         # Band pass filter
#         normal_cutoff_low = cutoff_low / nyq
#         normal_cutoff_high = cutoff_high / nyq
#         b, a = signal.butter(order, [normal_cutoff_low, normal_cutoff_high], btype='bandpass', analog=False)

#     filtered = signal.filtfilt(b, a, data)
#     return filtered
def butter_bandpass(data, fs, cutoff_low=0.05, cutoff_high=5, order=4):
    """
    Applica un filtro passa-banda Butterworth in forma stabile (sos) a un segnale.

    Parametri:
    - data: array 1D del segnale da filtrare
    - fs: frequenza di campionamento in Hz
    - cutoff_low: frequenza di taglio inferiore in Hz
    - cutoff_high: frequenza di taglio superiore in Hz
    - order: ordine del filtro (tipicamente 2-4)

    Ritorna:
    - filtered: segnale filtrato
    """
    nyq = 0.5 * fs  # Frequenza di Nyquist

    if cutoff_low == 0:
        # Filtro passa-basso
        normal_cutoff_high = cutoff_high / nyq
        sos = signal.butter(order, normal_cutoff_high, btype='low', output='sos')
    else:
        # Filtro passa-banda
        normal_cutoff_low = cutoff_low / nyq
        normal_cutoff_high = cutoff_high / nyq
        sos = signal.butter(order, [normal_cutoff_low, normal_cutoff_high], btype='bandpass', output='sos')

    filtered = signal.sosfiltfilt(sos, data)
    return filtered
# def pulse_segmentation(signal, waveOnset_list, peaks_list):
#     pulses = []
#     widths = []
#     amplitudes = []
#     peaks = []
#     for idx_min in range(0,(len(waveOnset_list)-1)):
#         if idx_min <= len(peaks_list)-1:
#             # if there is a peak between two local minima
#             if len(np.where((np.array(peaks_list)>waveOnset_list[idx_min])&(np.array(peaks_list)<waveOnset_list[idx_min+1]))[0])>0:
#                 pulse = signal[waveOnset_list[idx_min]:waveOnset_list[idx_min+1]]
#                 if len(pulse)> PWD_MIN and len(pulse) < PWD_MAX:
#                     pulse_width = len(pulse)
#                     peak = np.argmax(pulse)
#                     pulses.append(pulse)
#                     widths.append(pulse_width)
#                     peaks.append(peak)
#                     if waveOnset_list[idx_min] < peaks_list[idx_min]:
#                         amplitude = signal[peaks_list[idx_min]]- signal[waveOnset_list[idx_min]]
#                         amplitudes.append(amplitude)
#                     elif waveOnset_list[idx_min] > peaks_list[idx_min]:
#                         amplitude = signal[peaks_list[idx_min+1]]- signal[waveOnset_list[idx_min]]
#                         amplitudes.append(amplitude) 
#                 else:
#                     print(f"skip len")

#     segmentedPulses = pd.DataFrame({
#     'pulse': pulses,
#     'width': widths,
#     'amplitude': amplitudes,
#     'peak_pos': peaks})
#     return segmentedPulses


for path in dataPaths:
    if Path(path).suffix == ".npy":
        pleth = np.load(path)
    elif  Path(path).suffix == ".csv":
        pleth = pd.read_csv(path)["pleth"]
        pleth = np.array(pleth[::-1])
    
    filename = os.path.basename(path)
    time =  np.arange(0, len(pleth)) / SAMPLE_RATE
    outName = f"{filename[:-4]}_features.xlsx"
    # plt.plot(time, file)
    # plt.show()
    if REMOVE_DC:
        median = np.median(pleth)
        pleth = pleth - median
    if USE_FILTER:
        pleth = butter_bandpass(pleth.flatten(),SAMPLE_RATE,CUT_OFF_LOW, CUT_OFF_HIGH,ORDER)
        outName = f"{filename[:-4]}_filter{CUT_OFF_LOW}-{CUT_OFF_HIGH}-{ORDER}order_features.xlsx"
    if NORMALISE:
        pleth = sklearn.preprocessing.minmax_scale(pleth,(-1,1))
    
    # Create detector instance
    # FPD = fiducialPointsDetection(signal=pleth, time=time, fs=SAMPLE_RATE, df_cutoff_low=0.5, df_cutoff_high=3, df_order=1)
    # # Detect systolic peaks and valleys
    # sysPeaks = FPD.detect_systolic_peaks()
    # diastValleys = FPD.detect_valleys(sysPeaks)
    # finds systolic and diastolic peaks for pulse segmentation
    sysPeaks = FPD.detect_systolic_peaks(pleth,time,SAMPLE_RATE)#cutoff_low=0.5, cutoff_high=3, order=1)
    diastValleys = FPD.detect_valleys(pleth,time,sysPeaks)

    
    waveOnsets = diastValleys[:,0].astype(int)
    sysPoints = sysPeaks[:,0].astype(int)
    # coeffiecients of interpolant polynome
    tck = splrep(diastValleys[:, 1], diastValleys[:, 2], k=3)  # k=3 for cubic interpolation
    trend = splev(time, tck)
    # remove trend from signal 
    detrended_pleth = pleth - trend
    N = len(waveOnsets)
    featuresDict = {"pulse_index":[]}
    pulse_index = 0
    dic_notches = np.zeros((N, 3))
    # pulse segmentation
    for i in range(0, N - 1):
        pulse_index = pulse_index +1
        pulse = pleth[waveOnsets[i]:waveOnsets[i+1]]
        time_p = time[waveOnsets[i]:waveOnsets[i+1]]
        if len(pulse)< PWD_MIN or len(pulse) > PWD_MAX:
            print(f"skip pulse {pulse_index}")
            continue
        t_start = time_p[0]
        timePulse = np.arange(0, len(pulse)) / SAMPLE_RATE
        trend_pulse = trend[waveOnsets[i]:waveOnsets[i+1]]
        detrended_pulse = pulse-trend_pulse
    # pulse segmentation and features extraction on each beat 
    #segmentedPulsesDict = pulse_segmentation(detrended_pleth,waveOnsets,sysPoints)
    #segmentedPulses = segmentedPulsesDict["pulse"]
    # for pulseindex in range(len(segmentedPulses)):
    #     pulse = segmentedPulsesDict["pulse"].at[pulseindex]
        sysPosition = np.argmax(pulse)
        if sysPosition == 0:
            print(f"skip pulse {pulse_index}")
            #skip this PPG beat
            continue
        
        sysTime = timePulse[sysPosition]
        dicNotch, dicNotch_time = FPD.find_dicrotic_notch(np.array(detrended_pulse), timePulse, sysTime)
        dic_notches[i,0] = dicNotch
        dic_notches[i,1] = dicNotch_time + t_start
        dic_notches[i,2] = pulse[dicNotch]
        #perform feature extraction from PPG pulse
        featuresDict = MFE.extractRawPulseFeatures(pulse, timePulse, sysPosition, dicNotch, SAMPLE_RATE, featuresDict, plot=DEBUGGER_PLOTS)
        featuresDict = MFE.extractFirstDerivativePulseFeatures(pulse, timePulse, sysPosition, dicNotch, SAMPLE_RATE, featuresDict, plot=DEBUGGER_PLOTS)
        featuresDict = MFE.extractSecondDerivativePulseFeatures(pulse, timePulse, sysPosition, dicNotch, SAMPLE_RATE, featuresDict, plot=DEBUGGER_PLOTS)
        featuresDict["pulse_index"].append(pulse_index)
    
    #if DEBUGGER_PLOTS:
    plt.figure()
    plt.title(f"{filename}")
    plt.plot(time, pleth)
    plt.scatter(sysPeaks[:,1],pleth[sysPeaks[:,0].astype(int)],color="red", marker="*")
    plt.scatter(diastValleys[:,1],pleth[diastValleys[:,0].astype(int)],color="green")
    plt.scatter(dic_notches[:,1],dic_notches[:,2],color="cyan", marker="^")
    plt.show()
        
    featuresPerAcquisition = pd.DataFrame(featuresDict)
    featuresPerAcquisition.insert(0, "filename", filename)
    featuresPerAcquisition.to_excel(os.path.join(outDir,outName),index=False)
    