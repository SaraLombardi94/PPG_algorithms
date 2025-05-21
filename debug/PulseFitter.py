# This Python file uses the following encoding: utf-8
import numpy as np
from scipy.signal import argrelmax, argrelmin, argrelextrema
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

class PulseFitter:
    def __init__(self):
        pass
    
    @classmethod
    def skew_gaus_pulse(self, x, ti, ai, bi, alpha):
        sign = np.sign(x - ti)  # +1 per x > t, -1 per x < t
        # modify width based on asimmetry
        skew_factor = (1 + alpha * sign)
        return  ai * np.exp(-((x - ti) / (bi * skew_factor))**2)
    
    @classmethod
    def skew_gaussian_function(self, x, t1, a1, b1, alpha1, t2, a2, b2, alpha2, t3, a3, b3, alpha3):
        funct = (self.skew_gaus_pulse(x, t1, a1, b1, alpha1) +
                 self.skew_gaus_pulse(x, t2, a2, b2, alpha2) +
                 self.skew_gaus_pulse(x, t3, a3, b3, alpha3))
        return funct

    @classmethod
    def expPulse(self, t, t0, A, k1, k2):
        dt = t - t0
        v = 0.5 * (np.sign(dt) + 1) * A * (np.exp(-dt/k1) - np.exp(-dt/k2))
        return v
    @classmethod
    def exp_function(self, x, t1, a1, b1, c1, t2, a2, b2, c2, t3, a3, b3, c3):
        exp_functions = (
           self.expPulse(x, t1, a1, b1, c1) +
           self.expPulse(x, t2, a2, b2, c2) +
           self.expPulse(x, t3, a3, b3, c3))
        return exp_functions
    @classmethod
    def mse(self, y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)
    @classmethod
    def nrmse(self, y_true, y_pred, method="range"):
        rmse = np.sqrt(self.mse(y_true, y_pred))
        if method == "range":
            norm_factor = np.max(y_true) - np.min(y_true)
        elif method == "std":
            norm_factor = np.std(y_true)
        else:
            raise ValueError("Method must be 'range' or 'std'")

        return rmse / norm_factor if norm_factor != 0 else np.nan

    @classmethod
    def pulse_modelling(self, time_pulse, flux_pulse, dicnotch, fit_function):
        # define parameters to calculate model coefficients boundaries
        dicnotch_time, dicnotch_ampl = dicnotch[0], dicnotch[1]
        maxAmpl = np.max(flux_pulse) - np.min(flux_pulse)
        winSys = dicnotch_time - time_pulse[0]
        winRef = time_pulse[-1] - dicnotch[0]
        timeMax = time_pulse[np.argmax(flux_pulse)]
        tend = time_pulse[-1]
        minima_systole = argrelmin(flux_pulse[0:np.argmax(flux_pulse)])[0]
        if (minima_systole.size)>0:
            time_minima_systole = np.max(time_pulse[minima_systole])
        else:
            time_minima_systole = 0
        # initialises parameters depending on the function set in the UI (exp or gauss)
        if fit_function == "Exp":
            model_function = self.exp_function
            # ti -> wave starting point
            # ai -> max amplitude
            # bi -> first exp constant decay
            # ci -> second exp constant decay
            # first_wave
            t0, t0_lower, t0_upper = -timeMax/2, 0, timeMax/2
            a0, a0_lower, a0_upper = 2.5*maxAmpl,0.625*maxAmpl,  10*maxAmpl
            b0, b0_lower, b0_upper = winSys, winSys/4,winSys*2
            c0, c0_lower, c0_upper = timeMax/2, timeMax/8, timeMax*2
            # second_wave
            t1, t1_lower, t1_upper = 0, -timeMax/2, 2*timeMax
            a1, a1_lower, a1_upper = maxAmpl/2,  maxAmpl/8,  maxAmpl*2
            b1, b1_lower, b1_upper = winSys, winSys/4,  winSys*4
            c1, c1_lower, c1_upper = 5*tend, tend, tend*20
            # third_wave
            t2, t2_lower, t2_upper =  dicnotch_time, 0, dicnotch_time*4
            a2, a2_lower, a2_upper = 0.5*maxAmpl, 0, 2*maxAmpl
            b2, b2_lower, b2_upper = winRef*1.5, 3*winRef/8, winRef*6
            c2, c2_lower, c2_upper =  winRef/5, winRef/20, 4*winRef/2
            #                       t0         a0          b0         c0          t1         a1        b1        c1        t2       a2        b2          c2

            p0_lower = np.array([ t0_lower,  a0_lower,  b0_lower,  c0_lower,   t1_lower, a1_lower, b1_lower, c1_lower, t2_lower, a2_lower, b2_lower, c2_lower])
            p0_startpoint= np.array([ t0, a0, b0, c0, t1, a1, b1, c1, t2, a2, b2, c2])
            p0_upper = np.array([ t0_upper,  a0_upper,  b0_upper,  c0_upper,   t1_upper, a1_upper, b1_upper, c1_upper, t2_upper, a2_upper, b2_upper, c2_upper])

        elif fit_function =="Gauss":

            model_function = self.skew_gaussian_function
            minima_systole = argrelmin(flux_pulse[0:np.argmax(flux_pulse)])[0]
            if (minima_systole.size)>0:
                time_minima_systole = np.max(time_pulse[minima_systole])
            else:
                time_minima_systole = 0

            dicNotchIndex = np.where(time_pulse==dicnotch_time)[0][0]
            maxima_diastole = argrelmax(flux_pulse[dicNotchIndex:])[0]
            if (maxima_diastole.size)>0:
                time_maxima_diastole = np.min(time_pulse[maxima_diastole+dicNotchIndex])
            else:
                time_maxima_diastole = dicnotch_time+ winRef/4
            # first wave
            t1_lower, t1, t1_upper = timeMax/4, timeMax, timeMax*2
            a1_lower, a1, a1_upper = maxAmpl*0.5, maxAmpl*0.9, maxAmpl*1.5
            b1_lower, b1, b1_upper = timeMax/8, timeMax/4, timeMax
            alpha1_lower, alpha1, alpha1_upper = -2, 0, 2
            # second wave
            t2_lower, t2, t2_upper = timeMax, timeMax + (winSys-timeMax)/2, winSys
            a2_lower, a2, a2_upper = maxAmpl*0.01, maxAmpl*0.55, maxAmpl*0.7
            b2_lower, b2, b2_upper = (winSys-timeMax)/8, (winSys-timeMax)/4, (winSys-timeMax)/2
            alpha2_lower, alpha2, alpha2_upper = -2, 0, 2
            # third wave
            t3_lower, t3, t3_upper = winSys ,time_maxima_diastole, tend*0.9
            a3_lower, a3, a3_upper = maxAmpl*0.05, maxAmpl*0.2, maxAmpl*0.5
            b3_lower, b3, b3_upper = winRef/8, winRef/4, winRef/2
            alpha3_lower, alpha3, alpha3_upper = -2, 0, 2

            p0_lower      =  np.array([t1_lower , a1_lower, b1_lower, alpha1_lower,  t2_lower,  a2_lower , b2_lower , alpha2_lower, t3_lower , a3_lower ,   b3_lower, alpha3_lower])#, c_lower])
            p0_upper      =  np.array([ t1_upper , a1_upper, b1_upper, alpha1_upper, t2_upper,  a2_upper , b2_upper, alpha2_upper,  t3_upper, a3_upper, b3_upper, alpha3_upper])#, c_upper])
            p0_startpoint =  np.array([ t1, a1, b1, alpha1, t2, a2, b2, alpha2, t3, a3, b3, alpha3])

        # check if limits are appropriate for curve fitting
        if np.any(p0_lower > p0_startpoint):
            location = np.where(p0_lower > p0_startpoint)
            p0_lower[location] = p0_startpoint[location]

        if np.any(p0_upper < p0_startpoint):
            location = np.where(p0_upper < p0_startpoint)
            p0_upper[location] = p0_startpoint[location]

        if np.any(p0_lower>=p0_upper):
            location = np.where(p0_lower >= p0_upper)
            p0_upper[location] = p0_upper[location]+ p0_lower[location]
            #p0_lower = p0_lower/2
        try:
            model_params, cov = curve_fit(model_function, time_pulse, flux_pulse, p0=p0_startpoint, bounds=(p0_lower,p0_upper))
            r2 = r2_score(flux_pulse, model_function(time_pulse, *model_params))
            nrmse = self.nrmse(flux_pulse,model_function(time_pulse, *model_params))
            mse = self.mse(flux_pulse,model_function(time_pulse, *model_params))
            morph_params = {
            'winSys': winSys,
            'maxAmpl': maxAmpl,
            'timeSys': time_pulse[np.argmax(flux_pulse)] - time_pulse[0],
            'duration': tend - time_pulse[0]
            }
            errors = {
            'r2':r2,
            'nrmse':nrmse,
            'mse':mse
            }

        except Exception as e:
            errors = None
            model_params = None
            morph_params = None
            print(f"Could not fit file: {e}")

        return model_params, errors, morph_params

    @classmethod
    def fit_cycle(self,fluxCycle, timeCycle, trendCycle, dicnotch_time, fit_mode):

        model = np.zeros(len(fluxCycle))  # Pulse waves model
        waves = np.zeros((len(fluxCycle), 3))  # waves matrix
        direct = np.zeros(len(fluxCycle)) # direct wave
        reflex = np.zeros(len(fluxCycle)) # reflected waves

        fluxCycleDT = fluxCycle - trendCycle
        dicnotch_ampl = fluxCycleDT[timeCycle == dicnotch_time][0]
        dicNotch = [dicnotch_time,dicnotch_ampl]
        model_params, errors, morph_params = self.pulse_modelling(timeCycle,fluxCycleDT,dicNotch, fit_mode)
        if model_params is None:
            return None, None, None, None

        Parameters = {
        **morph_params,
        'dicnotch': dicnotch_time,
        'R2_of_fit': errors['r2'],
        'nrmse': errors['nrmse'],
        'mse': errors['mse']
        }

        for j in range(3):
            idx = j * 4 if fit_mode == "Exp" else j * 4
            t = model_params[idx]
            a = model_params[idx + 1]
            b = model_params[idx + 2]
            c_or_alpha = model_params[idx + 3]

            if fit_mode == "Exp":
                wave = self.expPulse(timeCycle, t, a, b, c_or_alpha)
                if j == 0:
                    direct += wave
                else:
                    reflex += wave
            elif fit_mode == "Gauss":
                wave = self.skew_gaus_pulse(timeCycle, t, a, b, c_or_alpha)
                if j == 0:
                    direct += wave
                else:
                    reflex += wave
            else:
                raise ValueError(f"Unsupported fit mode: {fit_mode}")

            waves[:, j] = wave
            model += wave

            # Salva parametri individuali
            Parameters[f't{j}'] = t
            Parameters[f'a{j}'] = a
            Parameters[f'b{j}'] = b
            Parameters[f'c{j}' if fit_mode == "Exp" else f'alpha{j}'] = c_or_alpha

        # energy metrics
        Parameters['Es'] = trapezoid(x=timeCycle, y=direct**2)
        Parameters['As'] = trapezoid(x=timeCycle, y=direct)
        Parameters['Ed'] = trapezoid(x=timeCycle, y=reflex**2)
        Parameters['Ad'] = trapezoid(x=timeCycle, y=reflex)
        Parameters['PulseAUC'] = trapezoid(x=timeCycle, y=fluxCycle)

        # trend metrics
        Parameters['td'] = trapezoid(x=timeCycle, y=reflex * timeCycle) / trapezoid(x=timeCycle, y=reflex)
        k = np.argmin(np.abs(timeCycle - Parameters['td']))
        Parameters['fd'] = trendCycle[k]

        return model, waves, Parameters, errors['r2']
