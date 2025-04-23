# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:56:42 2025

@author: adhk198
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import argrelmax, argrelmin, argrelextrema
from scipy.signal import find_peaks

def updateFeatureDict(dictionary, key, value):
    if key not in list(dictionary.keys()):
        dictionary[key] = []
        dictionary[key].append(value)
    elif key in list(dictionary.keys()):
        dictionary[key].append(value)
    return dictionary

def add_nan_features(featureDict, keys):
    for key in keys:
        featureDict = updateFeatureDict(featureDict, key, np.nan)
    return featureDict

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def findDiastolicPeak(time_win, signal_win, start_index):
    if np.max(signal_win) > signal_win[0]:
        maxIndex = np.argmax(signal_win)
        diastolicPeakIndex = start_index + maxIndex
    else:
        peaks,_ = find_peaks(signal_win,distance=10)
        if len(peaks)>0:
            diastolicPeakIndex = peaks[0] + start_index
        else:
            dfI = np.gradient(signal_win, time_win)
            maxIndex = np.where(dfI==np.max(dfI))[0]
            diastolicPeakIndex = start_index + maxIndex
    
    return diastolicPeakIndex
    

def extractRawPulseFeatures(single_waveform, time,  sysPeakInd, dicNotchInd, sample_rate, featureDict, plot=False):
    # systolic Peak time
    systolicPeakTime = time[sysPeakInd]
    featureDict = updateFeatureDict(featureDict, "t_sys", systolicPeakTime)
    
    # systolic Peak Amplitude
    systolicPeakAmplitude = single_waveform[sysPeakInd]
    featureDict = updateFeatureDict(featureDict, "sys_ampl", systolicPeakAmplitude)
    
    # dicNotch time and amplitude
    if np.isnan(dicNotchInd):
        dicNotchTime, dicNotchAmplitude = np.nan, np.nan
        featureDict = add_nan_features(featureDict, ["t_dn", "dicNotchAmpl"])
    else: 
        # dicNotch time
        dicNotchTime = time[dicNotchInd]
        featureDict = updateFeatureDict(featureDict, "t_dn", dicNotchTime)
        
        # dicNotch Amplitude
        dicNotchAmplitude = single_waveform[dicNotchInd]
        featureDict = updateFeatureDict(featureDict, "dn_ampl", dicNotchAmplitude)

    if np.isnan(dicNotchInd):
        win = single_waveform[sysPeakInd:int(len(single_waveform)*0.9)]
        time_win = time[sysPeakInd:int(len(single_waveform)*0.9)]
        diastolicPeakIndex = findDiastolicPeak(time_win,win,sysPeakInd)
    else:
        win = single_waveform[dicNotchInd:int(len(single_waveform)*0.9)]
        time_win = time[dicNotchInd:int(len(single_waveform)*0.9)]
        # find abs max after dicNotch
        diastolicPeakIndex = findDiastolicPeak(time_win,win,dicNotchInd)

        
        
    diastolicPeakTime = time[diastolicPeakIndex]
    featureDict = updateFeatureDict(featureDict, "t_dia", diastolicPeakTime)
        
    diastolicPeakAmplitude = single_waveform[diastolicPeakIndex]
    featureDict = updateFeatureDict(featureDict, "dia_ampl", diastolicPeakAmplitude)
    
    # augmentation index (AIx)
    AI = 100*((systolicPeakAmplitude-diastolicPeakAmplitude)/systolicPeakAmplitude)
    featureDict = updateFeatureDict(featureDict, "AI", AI)
    # reflection index (RI)
    RI = 100*(diastolicPeakAmplitude/systolicPeakAmplitude)
    featureDict = updateFeatureDict(featureDict, "RI", RI)
    
    # time from sys to dia
    ΔT = diastolicPeakTime - systolicPeakTime 
    featureDict = updateFeatureDict(featureDict, "ΔTsysdia", ΔT)
        
    # area from sys to diastolic rise
    xsw = time[np.where(time==systolicPeakTime)[0][0]:np.where(time==diastolicPeakTime)[0][0]]
    ysw = single_waveform[np.where(time==systolicPeakTime)[0][0]:np.where(time==diastolicPeakTime)[0][0]]
    areaSystoDiastPeak = np.trapz(ysw, x=xsw)
    featureDict = updateFeatureDict(featureDict, "area_sys_dia", areaSystoDiastPeak)
        
    # else:
    #     print('Could not find Diastolic Peak')
    #     diastolicPeakTime, diastolicPeakAmplitude, RI, AI, ΔT, areaSystoDiastPeak = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    #     featureDict = updateFeatureDict(featureDict, "diastolicPeakTime", diastolicPeakTime)
    #     featureDict = updateFeatureDict(featureDict, "diastolicPeakAmpl", diastolicPeakAmplitude)
    #     featureDict = updateFeatureDict(featureDict, "reflectionIndex", RI)
    #     featureDict = updateFeatureDict(featureDict, "augmentationIndex", AI)
    #     featureDict = updateFeatureDict(featureDict, "ΔTsysdia", ΔT)
    #     featureDict = updateFeatureDict(featureDict, "areaSysToDia", areaSystoDiastPeak)
    

    # pulse interval duration (from onset to onset)
    pulseDuration = time[-1]
    featureDict = updateFeatureDict(featureDict, "t_pulse", pulseDuration)
    
    # crest time ratio (CTR)
    ctr = systolicPeakTime/pulseDuration
    featureDict = updateFeatureDict(featureDict, "ctr", ctr)
    
    if np.isnan(dicNotchTime):
       featureDict = add_nan_features(featureDict, ["t_diastole"])
    else:
        # diastolic phase duration (from dicrotic notch)
        diastolicDuration = pulseDuration - dicNotchTime
        featureDict = updateFeatureDict(featureDict, "t_diastole", diastolicDuration)

    # width 10% of Amplitude
    amplitude10 = abs(systolicPeakAmplitude-np.min(single_waveform))*10/100
    index0_10 = find_nearest(single_waveform[0:sysPeakInd], (np.min(single_waveform)+amplitude10))
    index1_10 = sysPeakInd+(find_nearest(single_waveform[sysPeakInd:],(np.min(single_waveform)+amplitude10)))
    w10 = time[index1_10]- time[index0_10] 
    sw10 = systolicPeakTime - time[index0_10]
    dw10 = time[index1_10] - systolicPeakTime
    featureDict = updateFeatureDict(featureDict, "duration10%Ampl", w10)    
    featureDict = updateFeatureDict(featureDict, "sw10%Ampl", sw10)  
    featureDict = updateFeatureDict(featureDict, "dw10%Ampl", dw10)  
    
    
    # width 25% of Amplitude
    amplitude25 = abs(systolicPeakAmplitude-np.min(single_waveform))*25/100
    index0_25 = find_nearest(single_waveform[0:sysPeakInd], (np.min(single_waveform)+amplitude25))
    index1_25 = sysPeakInd+(find_nearest(single_waveform[sysPeakInd:],(np.min(single_waveform)+amplitude25)))
    w25 = time[index1_25]- time[index0_25] 
    sw25 = systolicPeakTime - time[index0_25]
    dw25 = time[index1_25] - systolicPeakTime
    featureDict = updateFeatureDict(featureDict, "duration25%Ampl", w25)
    featureDict = updateFeatureDict(featureDict, "sw25%Ampl", sw25)  
    featureDict = updateFeatureDict(featureDict, "dw25%Ampl", dw25)  

    # width 33% of Amplitude
    amplitude33 = abs(systolicPeakAmplitude-np.min(single_waveform))*33/100
    index0_33 = find_nearest(single_waveform[0:sysPeakInd], (np.min(single_waveform)+amplitude33))
    index1_33 = sysPeakInd+(find_nearest(single_waveform[sysPeakInd:],(np.min(single_waveform)+amplitude33)))
    w33 = time[index1_33]- time[index0_33] 
    sw33 = systolicPeakTime - time[index0_33]
    dw33 = time[index1_33] - systolicPeakTime
    featureDict = updateFeatureDict(featureDict, "duration33%Ampl", w33)
    featureDict = updateFeatureDict(featureDict, "sw33%Ampl", sw33)  
    featureDict = updateFeatureDict(featureDict, "dw33%Ampl", dw33)  
    
    # width 50% of Amplitude
    amplitude50 = abs(systolicPeakAmplitude-np.min(single_waveform))*50/100
    index0_50 = find_nearest(single_waveform[0:sysPeakInd], (np.min(single_waveform)+amplitude50))
    index1_50 = sysPeakInd+(find_nearest(single_waveform[sysPeakInd:],(np.min(single_waveform)+amplitude50)))
    w50 = time[index1_50]- time[index0_50] 
    sw50 = systolicPeakTime - time[index0_50]
    dw50 = time[index1_50] - systolicPeakTime
    featureDict = updateFeatureDict(featureDict, "duration50%Ampl", w50)
    featureDict = updateFeatureDict(featureDict, "sw50%Ampl", sw50)  
    featureDict = updateFeatureDict(featureDict, "dw50%Ampl", dw50)  

    # width 66% of Amplitude
    amplitude66 = abs(systolicPeakAmplitude-np.min(single_waveform))*66/100
    index0_66 = find_nearest(single_waveform[0:sysPeakInd], (np.min(single_waveform)+amplitude66))
    index1_66 = sysPeakInd+(find_nearest(single_waveform[sysPeakInd:],(np.min(single_waveform)+amplitude66)))
    w66 = time[index1_66]- time[index0_66] 
    sw66 = systolicPeakTime - time[index0_66]
    dw66 = time[index1_66] - systolicPeakTime
    featureDict = updateFeatureDict(featureDict, "duration66%Ampl", w66)
    featureDict = updateFeatureDict(featureDict, "sw66%Ampl", sw66)  
    featureDict = updateFeatureDict(featureDict, "dw66%Ampl", dw66)  
    
    # width 75% of Amplitude
    amplitude75 = abs(systolicPeakAmplitude-np.min(single_waveform))*75/100
    index0_75 = find_nearest(single_waveform[0:sysPeakInd], (np.min(single_waveform)+amplitude75))
    index1_75 = sysPeakInd+(find_nearest(single_waveform[sysPeakInd:],(np.min(single_waveform)+amplitude75)))
    w75 = time[index1_75]- time[index0_75] 
    sw75 = systolicPeakTime - time[index0_75]
    dw75 = time[index1_75] - systolicPeakTime
    featureDict = updateFeatureDict(featureDict, "duration75%Ampl", w75)
    featureDict = updateFeatureDict(featureDict, "sw75%Ampl", sw75)  
    featureDict = updateFeatureDict(featureDict, "dw75%Ampl", dw75) 
    
    # area under the pulse wave:
    areaPulse = np.trapz(single_waveform, x=time)
    featureDict = updateFeatureDict(featureDict, "AUC", areaPulse)
    
    # area SystolicPeak 
    xsw = time[:np.where(time==systolicPeakTime)[0][0]]
    ysw = single_waveform[:np.where(time==systolicPeakTime)[0][0]]
    areaSysPeak = np.trapz(ysw, x=xsw)
    featureDict = updateFeatureDict(featureDict, "area_sysPeak", areaSysPeak)
    
    if np.isnan(dicNotchTime):
        add_nan_features(featureDict, ["area_sys", "area_dia", "ratioAsys_Adn"])
    else:
        # areaSw = area under systolic waveform
        xsw = time[:np.where(time==dicNotchTime)[0][0]]
        ysw = single_waveform[:np.where(time==dicNotchTime)[0][0]]
        areaSw = np.trapz(ysw, x=xsw)
    
    
        # areaDw = area under diastolic waveform
        xdw = time[np.where(time==dicNotchTime)[0][0]:]
        ydw = single_waveform[np.where(time==dicNotchTime)[0][0]:]
        areaDw = np.trapz(ydw, x = xdw)
        # areaDw/areaSw
        ratioAswAdw = areaDw/areaSw
            
        featureDict = updateFeatureDict(featureDict, "area_sys", areaSw)
        featureDict = updateFeatureDict(featureDict, "area_dn", areaDw)
        featureDict = updateFeatureDict(featureDict, "ratioAsys_Adn", ratioAswAdw)
    
    if plot:
        # Plot Creation
        plt.figure()
        plt.plot(time,single_waveform,color='blue')
        plt.scatter(systolicPeakTime, systolicPeakAmplitude, color='darkorange', label='Systolic Peak', marker="o")
        plt.scatter(dicNotchTime, dicNotchAmplitude, color='green', label='Dicrotic Notch', marker="o")
        #plt.axvline(x=dicNotchTime,ymax=dicNotchAmplitude, color='darkred', linestyle='--', linewidth=1)
        plt.scatter(diastolicPeakTime, diastolicPeakAmplitude, color= 'purple',label='Diastolic Peak', marker="o")
        plt.scatter(time[index0_10],single_waveform[index0_10], color='brown', label='Width10%', marker="_")
        plt.scatter(time[index1_10],single_waveform[index1_10], color='brown',marker="_")
        plt.scatter(time[index0_25],single_waveform[index0_25], color='maroon', label='Width25%', marker="_")
        plt.scatter(time[index1_25],single_waveform[index1_25], color='maroon',marker="_")
        plt.scatter(time[index0_33],single_waveform[index0_33], color='indigo', label='Width33%', marker="_")
        plt.scatter(time[index1_33],single_waveform[index1_33], color='indigo',marker="_")
        plt.scatter(time[index0_50],single_waveform[index0_50], color='black', label='Width50%', marker="_")
        plt.scatter(time[index1_50],single_waveform[index1_50], color='black',marker="_")
        plt.scatter(time[index0_66],single_waveform[index0_66], color='darkslategrey', label='Width50%', marker="_")
        plt.scatter(time[index1_66],single_waveform[index1_66], color='darkslategrey',marker="_")
        plt.scatter(time[index0_75],single_waveform[index0_75], color='saddlebrown', label='Width75%', marker="_")
        plt.scatter(time[index1_75],single_waveform[index1_75], color='saddlebrown',marker="_")
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize="small")
        plt.tight_layout()
        plt.show()
    return featureDict


def extractFirstDerivativePulseFeatures(single_waveform, time, sysPosition, dicNotch, sample_rate, featureDict, plot=False):
    # First derivative
    derivative1 = np.gradient(single_waveform, time)
    #derivative1 = np.convolve(derivative1, np.ones(3)/3, mode='same')
    
    # search u peak as abs maximum occurring before systolic peak time
    window_u_peak = derivative1[:sysPosition]
    u_peak_idx = np.argmax(np.abs(window_u_peak)) 
    u_peak_time = time[u_peak_idx]
    u_peak = derivative1[u_peak_idx]
    featureDict = updateFeatureDict(featureDict, "t_uPeak",  u_peak_time)
    featureDict = updateFeatureDict(featureDict, "uPeak",  u_peak)
    
    # search v peak as abs minimum occurring after systolic peak 
    window_v_peak = derivative1[sysPosition:]
    v_peak_idx = np.argmin(window_v_peak) + sysPosition
    v_peak_time = time[v_peak_idx]
    v_peak = derivative1[v_peak_idx]
    featureDict = updateFeatureDict(featureDict, "t_vPeak",  v_peak_time)
    featureDict = updateFeatureDict(featureDict, "vPeak",  v_peak)
    
    t_u_v = v_peak_time - u_peak_time
    featureDict = updateFeatureDict(featureDict, "time_u_v",  t_u_v)
    
    # search w peak as abs maximum occurring after dicrotich notch
    window_w_peak = derivative1[dicNotch:int(len(derivative1)*0.8)]
    w_peak_idx = np.argmax(window_w_peak) + dicNotch
    w_peak_time = time[w_peak_idx]
    w_peak = derivative1[w_peak_idx]

    featureDict = updateFeatureDict(featureDict, "t_wPeak",  w_peak_time)
    featureDict = updateFeatureDict(featureDict, "wPeak",  w_peak)
    
    
    t_u_pulse_ratio = u_peak_time/time[-1]
    t_v_pulse_ratio = v_peak_time/time[-1]
    t_w_pulse_ratio = w_peak_time/time[-1]
    featureDict = updateFeatureDict(featureDict, "t_u_pulse",  t_u_pulse_ratio)
    featureDict = updateFeatureDict(featureDict, "t_v_pulse",  t_v_pulse_ratio)
    featureDict = updateFeatureDict(featureDict, "t_w_pulse",  t_w_pulse_ratio)

    # if PLOT:
    #     plt.figure()
    #     plt.plot(time, derivative1, color='orange', label='First Derivative')
    #     plt.scatter(u_peak_time, u_peak, color='red',marker='*', label='uPoint')
    #     plt.scatter(v_peak_time, v_peak, color='darkmagenta',marker='*', label='vPoint')
    #     #plt.axvline(x=systMaxTime, ymin=0, ymax=systMaxAmpl,  color="black", linestyle='--', linewidth=0.5)
    #     #plt.axhline(y= derivative1[zeroIndex],color="black", linestyle='--', linewidth=0.4)
    #     #plt.axvline(x=wPeakTime,ymin=0, ymax=wPeakAmplitude, color="black", linestyle='--', linewidth=0.5)
    #     plt.scatter(w_peak_time, w_peak, color= 'blue', marker='*', label = 'wPoint')
    #     plt.xlabel('Time')
    #     plt.ylabel('Amplitude')
    #     plt.legend(loc='upper right', fontsize = "small")
    #     plt.show()
        
    # return featureDict
    if plot:
        plt.figure()
        plt.plot(time, derivative1, color='orange', label='First Derivative')
        plt.scatter(u_peak_time, u_peak, color='red', marker='*', label='uPoint')
        plt.scatter(v_peak_time, v_peak, color='darkmagenta', marker='*', label='vPoint')
        #plt.axvline(x=systMaxTime, ymin=0, ymax=systMaxAmpl,  color="black", linestyle='--', linewidth=0.5)
        #plt.axhline(y= derivative1[zeroIndex],color="black", linestyle='--', linewidth=0.4)
        #plt.axvline(x=wPeakTime,ymin=0, ymax=wPeakAmplitude, color="black", linestyle='--', linewidth=0.5)
        plt.scatter(w_peak_time, w_peak, color='blue', marker='*', label='wPoint')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize="small")
        plt.tight_layout()
        plt.show()
    
    return featureDict

def extractSecondDerivativePulseFeatures(single_waveform, time,  sysPosition, dicNotch, sample_rate, featureDict, plot=False):
    # TODO: find a as the abs max before systolic peak, 
    
    
    #find d as max point between systolic peak and dicrotic notch

    # Second derivative
    derivative1 = np.gradient(single_waveform, time)
    derivative2 = np.gradient(derivative1, time)
    
    # Search A point as the abs max before systolic peak, 
    a_window = derivative2[:sysPosition]
    a_peak_idx = np.argmax(a_window) 
    a_peak_time = time[a_peak_idx]
    a_peak = derivative2[a_peak_idx]
    ppg_apoint = single_waveform[a_peak_idx]
    featureDict = updateFeatureDict(featureDict, "t_Apoint",  a_peak_time)
    featureDict = updateFeatureDict(featureDict, "APoint",  a_peak)
    
    # Search B as absolute min before systolic peak
    b_peak_idx = np.argmin(a_window) 
    b_peak_time = time[b_peak_idx]
    b_peak = derivative2[b_peak_idx]
    ppg_bpoint = single_waveform[b_peak_idx]
    featureDict = updateFeatureDict(featureDict, "t_Bpoint",  b_peak_time)
    featureDict = updateFeatureDict(featureDict, "BPoint",  b_peak)
    
    
    # Search E as max point after systolic peak 
    e_window = derivative2[sysPosition:int(len(single_waveform)*0.9)]
    e_peak_idx = np.argmax(e_window) + sysPosition
    e_peak_time = time[e_peak_idx]
    e_peak = derivative2[e_peak_idx]
    ppg_epoint = single_waveform[e_peak_idx]
    featureDict = updateFeatureDict(featureDict, "t_Epoint",  e_peak_time)
    featureDict = updateFeatureDict(featureDict, "EPoint",  e_peak)
    featureDict = updateFeatureDict(featureDict,"PPGAmpl_Epoint", ppg_epoint)
    
    # Search C as max point between systolic peak and E point
    c_window = derivative2[b_peak_idx:e_peak_idx]
    idx,_ = find_peaks(c_window)
    if len(idx)>0:
        idx = idx + b_peak_idx
        pos_peak_values = derivative2[idx]
        max_index = np.argmax(pos_peak_values)
        c_peak_idx = idx[max_index] 
        c_peak_time = time[c_peak_idx]
        c_peak = derivative2[c_peak_idx]
        featureDict = updateFeatureDict(featureDict, "t_Cpoint",  c_peak_time)
        featureDict = updateFeatureDict(featureDict, "CPoint",  c_peak) 
        # c/a
        ratio_ca = c_peak/a_peak
        featureDict = updateFeatureDict(featureDict,"ratio_C-A", ratio_ca)
    else:
        c_peak_idx, c_peak_time, c_peak, ratio_ca = np.nan, np.nan, np.nan, np.nan
        featureDict = updateFeatureDict(featureDict, "t_Cpoint",  c_peak_time)
        featureDict = updateFeatureDict(featureDict, "CPoint",  c_peak) 
        featureDict = updateFeatureDict(featureDict,"ratio_C-A", ratio_ca)
    
    if not np.isnan(c_peak_idx): 
        # Search D as min point between C point and E Point
        d_window = derivative2[c_peak_idx:e_peak_idx]
        neg_idx,_ = find_peaks(-d_window)
        if len(neg_idx)>0:
            neg_idx = neg_idx  + c_peak_idx
            neg_peak_values = derivative2[neg_idx]
            min_index = np.argmin(neg_peak_values) 
            d_peak_idx = neg_idx[min_index] 
            d_peak_time = time[d_peak_idx]
            d_peak = derivative2[d_peak_idx]
            featureDict = updateFeatureDict(featureDict, "t_Dpoint",  d_peak_time)
            featureDict = updateFeatureDict(featureDict, "DPoint",  d_peak)
            # d/a
            ratio_da = d_peak/a_peak
            featureDict = updateFeatureDict(featureDict,"ratio_D-A", ratio_da)
        else:
            d_peak_time, d_peak, ratio_da = np.nan, np.nan, np.nan
            featureDict = updateFeatureDict(featureDict, "t_Dpoint",  d_peak_time)
            featureDict = updateFeatureDict(featureDict, "DPoint",  d_peak)
            featureDict = updateFeatureDict(featureDict,"ratio_D-A", ratio_da)
    else:
        d_peak_time, d_peak, ratio_da = np.nan, np.nan, np.nan
        featureDict = updateFeatureDict(featureDict, "t_Dpoint",  d_peak_time)
        featureDict = updateFeatureDict(featureDict, "DPoint",  d_peak)
        featureDict = updateFeatureDict(featureDict,"ratio_D-A", ratio_da)
    
    # PPG'' tot ampl
    ampl_ba = abs(b_peak-a_peak)
    featureDict = updateFeatureDict(featureDict,"PPG''_tot_ampl", ampl_ba)
    # b/a
    ratio_ba = b_peak/a_peak
    featureDict = updateFeatureDict(featureDict,"ratio_B-A", ratio_ba)


    # PPG[B]/PPG[A]
    ratio_ppg_ba = ppg_bpoint/ppg_apoint
    featureDict = updateFeatureDict(featureDict,"ratio_PPG[B]_PPG[A]", ratio_ppg_ba)
    # e/a
    ratio_ea = e_peak/a_peak
    featureDict = updateFeatureDict(featureDict,"ratio_E-A", ratio_ea)
    # PPG[E]/PPG[A]
    ratio_ppg_ea = ppg_epoint/ppg_apoint
    featureDict = updateFeatureDict(featureDict,"ratio_PPG[E]_PPG[A]", ratio_ppg_ea)
    #
    if plot:
        plt.figure()
        plt.plot(time, derivative2, color='teal', label='Second Derivative')
        plt.scatter(a_peak_time, a_peak, color='black', label='A_point')
        plt.scatter(b_peak_time, b_peak, color='navy', label='B_point')
        plt.scatter(c_peak_time, c_peak, color='darkslategray', label='C_point')
        plt.scatter(d_peak_time, d_peak, color='darkviolet', label='D_point')
        plt.scatter(e_peak_time, e_peak, color='crimson', label='E_point')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize="small")
        plt.tight_layout()
        plt.show()
    return featureDict