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
from fiducialPointsDetection import fiducialPointsDetection as FPD
import morphologicalFeatureExtraction as MFE
from pathlib import Path



dataDir = r"D:\PPGFilteringProject\in-vivo-data\dataArray\finger\array"
outDirFeatures = r"D:\PPGFilteringProject\extractedFeatures"
dataPaths = glob(os.path.join(dataDir,"*PPG_IR*"))
SAMPLE_RATE = 2000 #Hz

# duration of systolic phase 
SP_MIN = 0.08*SAMPLE_RATE
SP_MAX = 0.49*SAMPLE_RATE
#physiological parameters
HR_MIN = 50 #bpm
HR_MAX = 100 #bpm
# min and max pulse duration 
PWD_MIN = round((60/HR_MAX)*SAMPLE_RATE) #samples
PWD_MAX = round((60/HR_MIN)*SAMPLE_RATE)

USE_FILTER = True
REMOVE_DC = False
NORMALISE = False
#filter parameters
CUT_OFF_LOW = 0.05
CUT_OFF_HIGH = 20
ORDER = 2
DEBUGGER_PLOTS = True

def butter_bandpass(data,fs, cutoff_low=0.05, cutoff_high=10, order=2):        
    nyq = 0.5 * fs
    normal_cutoff_low = cutoff_low / nyq
    normal_cutoff_high = cutoff_high / nyq
    b, a = signal.butter(order, [normal_cutoff_low, normal_cutoff_high], btype='bandpass', analog=False)
    filtered = signal.filtfilt(b, a, data)
    return filtered

def pulse_segmentation(signal, waveOnset_list, peaks_list):
    pulses = []
    widths = []
    amplitudes = []
    peaks = []
    for idx_min in range(0,(len(waveOnset_list)-1)):
        if idx_min <= len(peaks_list)-1:
            # if there is a peak between two local minima
            if len(np.where((np.array(peaks_list)>waveOnset_list[idx_min])&(np.array(peaks_list)<waveOnset_list[idx_min+1]))[0])>0:
                pulse = signal[waveOnset_list[idx_min]:waveOnset_list[idx_min+1]]
                if len(pulse)> PWD_MIN and len(pulse) < PWD_MAX:
                    pulse_width = len(pulse)
                    peak = np.argmax(pulse)
                    pulses.append(pulse)
                    widths.append(pulse_width)
                    peaks.append(peak)
                    if waveOnset_list[idx_min] < peaks_list[idx_min]:
                        amplitude = signal[peaks_list[idx_min]]- signal[waveOnset_list[idx_min]]
                        amplitudes.append(amplitude)
                    elif waveOnset_list[idx_min] > peaks_list[idx_min]:
                        amplitude = signal[peaks_list[idx_min+1]]- signal[waveOnset_list[idx_min]]
                        amplitudes.append(amplitude) 

    segmentedPulses = pd.DataFrame({
    'pulse': pulses,
    'width': widths,
    'amplitude': amplitudes,
    'peak_pos': peaks})
    return segmentedPulses


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
    
    # finds systolic and diastolic peaks for pulse segmentation
    sysPeaks = FPD.sysDet(pleth,time,SAMPLE_RATE,cutoff_low=0.5, cutoff_high=3, order=1)
    diastValleys = FPD.valDet(pleth,time,sysPeaks)
    if DEBUGGER_PLOTS:
        plt.figure()
        plt.title(f"{filename}")
        plt.plot(pleth)
        plt.scatter(sysPeaks[:,0].astype(int),pleth[sysPeaks[:,0].astype(int)],color="red")
        plt.scatter(diastValleys[:,0].astype(int),pleth[diastValleys[:,0].astype(int)],color="green")
        plt.show()
    
    waveOnsets = diastValleys[:,0].astype(int)
    sysPoints = sysPeaks[:,0].astype(int)
    # coeffiecients of interpolant polynome
    tck = splrep(diastValleys[:, 1], diastValleys[:, 2], k=3)  # k=3 for cubic interpolation
    trend = splev(time, tck)
    # remove trend from signal 
    detrended_pleth = pleth - trend
    # HRV features extraction
    # hrv_features = HRVE.HRVFeatureExtractor(detrended_pleth,sysPoints,SAMPLE_RATE)
    # hrv_features = pd.DataFrame(hrv_features)
    # hrv_features.insert(0, "filename", filename)
    # hrv_features.to_excel(os.path.join(outDirFeatures,f"{filename}_hrv.xlsx"),index=False)
    
    # pulse segmentation and features extraction on each beat 
    segmentedPulsesDict = pulse_segmentation(detrended_pleth,waveOnsets,sysPoints)
    segmentedPulses = segmentedPulsesDict["pulse"]
    featuresDict = {}
    for pulseindex in range(len(segmentedPulses)):
        pulse = segmentedPulsesDict["pulse"].at[pulseindex]
        sysPosition = segmentedPulsesDict["peak_pos"].at[pulseindex]
        timePulse = np.arange(0, len(pulse)) / SAMPLE_RATE
        sysTime = timePulse[sysPosition]
        dicNotch, dicNotch_time = FPD.dicNotchDetect(np.array(pulse), timePulse, sysTime, SAMPLE_RATE,plot=DEBUGGER_PLOTS)

        # perform feature extraction from PPG pulse
        featuresDict = MFE.extractRawPulseFeatures(pulse, timePulse, sysPosition, dicNotch, SAMPLE_RATE, featuresDict, plot=DEBUGGER_PLOTS)
        featuresDict = MFE.extractFirstDerivativePulseFeatures(pulse, timePulse, sysPosition, dicNotch, SAMPLE_RATE, featuresDict, plot=False)
        featuresDict = MFE.extractSecondDerivativePulseFeatures(pulse, timePulse, sysPosition, dicNotch, SAMPLE_RATE, featuresDict, plot=False)
        
    featuresPerAcquisition = pd.DataFrame(featuresDict)
    featuresPerAcquisition.insert(0, "filename", filename)
    featuresPerAcquisition.to_excel(os.path.join(outDirFeatures,outName),index=False)
    