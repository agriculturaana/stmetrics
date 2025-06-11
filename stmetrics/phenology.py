import numpy
import matplotlib.pyplot as plt


def get_filtered_series(timeseries,window,treshold):
    
    """
    
    This function filter the input timeseries using the Savitzky-Golay method, using \\
    a similar approach to the Timesat. 
    
    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.
        treshold : float
            Minimum growing that will be used to detect a new cycle.
        window: integer (odd number) default 7
            Size of the window used for filtering with the Savitzky-Golay 
    Returns
    -------
    numpy.ndarray:
        filtered timeseries (sg)
    
    """
    from scipy.signal import savgol_filter

    min_win = int(numpy.floor(window/2))
    y = numpy.empty([min_win, timeseries.shape[0]])
    
    pos = 0
    for i in range(window,2,-2):
        y[pos,:] = savgol_filter(timeseries,i,2)
        #plt.plot(y[pos,:])
        pos+=1

    diff = abs(y[0]/timeseries)
    diff[numpy.isnan(diff)]=0
    diff[numpy.isinf(diff)]=0
    
    max_diff = abs(numpy.max(diff))

    level = 0
    sg = y[0]

    while max_diff >= treshold and level < pos-1:
        new = y[level+1,:]
        sg = numpy.where(diff>treshold,new,sg)
        diff = abs(sg/timeseries)
        diff[numpy.isnan(diff)]=0
        diff[numpy.isinf(diff)]=0
        max_diff = abs(numpy.max(diff))-1
        level += 1
    
    return sg

def get_ts_metrics(y,a,c,l_mi,b,d,r_mi,tresh):

    """
    
    This function compute the phenological metrics
    
    Reference: 

    Keyword arguments:
       
    Returns
    -------
    numpy.ndarray:
        array of peaks
    """
    from sklearn.metrics import auc

    #Interpolate a, b, c and d positions
    #Fine adjustment
    xp = numpy.arange(0,len(y))
    start_val = numpy.interp(a, xp, y)
    end_val = numpy.interp(b, xp, y)
    yc = numpy.interp(c, xp, y)
    yd = numpy.interp(d, xp, y)
    
    #Compute other metrics
    l_derivada = abs(start_val-yc)/abs(a-c)
    r_derivada = abs(end_val-yd)/abs(b-d)  
    lenght = abs(a-b)
    Base_val = numpy.array([l_mi,r_mi]).mean()
    Peak_val = numpy.max(y[int(a):int(b)])
    Peak_t = list(y[int(a):int(b)]).index(Peak_val)+a
    ampl = Peak_val - Base_val

    #compute areas
    xx = numpy.arange(int(a),int(b),1)
    yy = y[int(a):int(b)]
    yy2 = abs(y[int(a):int(b)])+abs(Base_val)
    h = auc(xx,yy)
    i = auc(xx,yy2)
    
    return numpy.array([a,b,lenght,Base_val,Peak_t,Peak_val,ampl,l_derivada,r_derivada,h,i,start_val,end_val])

def get_greenness(series,start,midle,minimum_up):


    """
    
    This function get info about the browness of a cycle.
    
    Reference: 

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.
        start : integer 
            Position along time series axis
        minimum_up : float 
            Minimum growing that will be used to detect a new cycle.

    Returns
    -------
    numpy.ndarray:
        array of peaks
    """
    
    series = series[start:midle]   

    #funtion to get the start/end of interval
    idx = numpy.where(series == numpy.amin(series))[0][0]
    series = series[idx:]
    l_mi = min(series)
    ma = max(series)
    dis = ma-l_mi

    #get cummulative sum from left side
    ret = ((series - l_mi)/ma)
    xp = numpy.arange(0,len(ret))    
    c = numpy.interp(1-minimum_up, ret, xp) + start 
    a = numpy.interp(minimum_up, ret, xp) + start

    return a,c,l_mi

def get_brownness(series,midle,end,minimum_up):

    """
    
    This function get info about the browness of a cycle.
    
    Reference: 

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.
        start : integer 
            Position along time series axis
        minimum_up : integer 
            Minimum growing that will be used to detect a new cycle.

    Returns
    -------
    numpy.ndarray:
        array of peaks
    """
    
    series = series[midle:end]
    
    #funtion to get the start/end of interval  
    idx = numpy.where(series == numpy.amin(series))[0][0]
    series = series[:idx]
    series = series[::-1]
    #print(idx,series)
    r_mi = min(series)
    ma = max(series)
    dis = ma-r_mi

    #get cummulative sum from left side    
    ret = (series - r_mi)/ma
    xp = numpy.arange(0,len(ret))    

    d = midle + idx - numpy.interp(1-minimum_up, ret, xp) + 1
    b = midle + idx - numpy.interp(minimum_up, ret, xp)
    
    return b,d,r_mi

def get_peaks(timeseries,minimum_up):
    """
    
    This function find the peaks of a time series.
    
    Reference: 

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.
        minimum_up : float
            Minimum growing that will be used to detect a new cycle.
    Returns
    -------
    numpy.ndarray:
        array of peaks
    """
    from scipy.signal import savgol_filter,medfilt,find_peaks

    b = (numpy.diff(numpy.sign(numpy.diff(timeseries))) > 0).nonzero()[0] +1# local min
    c = (numpy.diff(numpy.sign(numpy.diff(timeseries))) < 0).nonzero()[0] +1# local max
    
    picos = numpy.sort(numpy.asarray(b.tolist()+c.tolist())).tolist()
    picos.append(1)
    picos = sorted(picos)
    picos.append(timeseries.shape[0]-1)
    
    peaks = []
    px = picos.copy()
    
    cond = False
    peak = 0

    while cond != True:
        if timeseries[picos[peak]] < timeseries[picos[peak+1]]:
            cond = True
            peak = len(picos)+1
        else:
            px.remove(picos[peak])
            peak=peak+1
    
    picos = px

    for peak in range(1,len(picos)-1,2):
        if timeseries[picos[peak-1]] < timeseries[picos[peak]] and abs((timeseries[picos[peak-1]]/timeseries[picos[peak]])-1)>=minimum_up or  timeseries[picos[peak]] > timeseries[picos[peak+1]] and abs((timeseries[picos[peak]]/timeseries[picos[peak+1]]) - 1)>=minimum_up:
            peaks.append(picos[peak-1])
            peaks.append(picos[peak])
            peaks.append(picos[peak+1])
    
    return numpy.asarray(peaks)


def domain(y,peaks,minimum_up,min_height):

    """
    
    This function find the cycles inside a timeseries.
    
    Reference: 

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.
        peaks : list
            Peaks that are detected using function get_peaks.
        minimum_up : float
            Minimum growing that will be used to detect a new cycle.

    Returns
    -------
    pandas.dataframe:
        dataframe with the phenometrics
    """
    import pandas
    
    header_timesat=["Start","End","Length","Base val.","Peak t.","Peak val.","Ampl.","L. Deriv.","R. Deriv.","S.integral","L.integral","Start val.","End val."]
    dfp = pandas.DataFrame(columns=header_timesat)
    
    pos = 0
    for peak in range(1,len(peaks),2):
        midle = None
        end = None
        lpeak = peaks[peak:]
        start = peaks[peak-1]
        
        try:
            cond_midle = False
            pp = peak
            while cond_midle != True and pp < (peaks.shape[0]-1):
                if y[peaks[pp]]-y[start] >= min_height :
                    cond_midle = True
                    midle=pp
                    pp = peaks.shape[0]
                else:
                    pp=pp+1
        except:
            continue

        try:
            cond = False
            pp = midle+1
            while cond != True or pp < peaks.shape[0]-1:
                if y[peaks[midle]]-y[peaks[pp]] >= min_height:
                    cond = True
                    end=pp
                    pp = peaks.shape[0]
                else:
                    pp=pp+1
        except:
            continue
        
        try:
            pmidle = peaks[midle]
            pend =  peaks[end]
        except:
            continue

        if y[start:pmidle].shape[0] <= 2:
            continue
        
        # local min
        vales = (numpy.diff(numpy.sign(numpy.diff(y[start:pmidle]))) > 0).nonzero()[0] +1
        # local max
        picos = (numpy.diff(numpy.sign(numpy.diff(y[start:pmidle]))) < 0).nonzero()[0] +1
        
        if len(vales) > 0 or len(picos) > 0:
            continue
       
        try:
            #get left side
            a,c,l_mi = get_greenness(y,start,pmidle,minimum_up)
        except:
            continue
       
        #get right side    
        if y[pmidle:pend].shape[0] <= 2:
            continue
        try:
            b,d,r_mi = get_brownness(y,pmidle,pend,minimum_up)
        except:
            continue    

        dfp.loc[pos] = get_ts_metrics(y,a,c,l_mi,b,d,r_mi,minimum_up)
        dfp = dfp.dropna()        

        pos +=1
                    
    return dfp

def phenometrics(time_series,  minimum_up= 0.05, min_height = 0.1, 
                smooth_fraction=0.2, periods=24,  treshold = 0.125,
                window = 7, show=False, iterations=3):
    """
    This function computes phenological metrics.
    
    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.
        minimum_up : float
            Minimum growing that will be used to detect a new cycle.
        min_height: float, 0.1 default
            Minimum heigth of cycle to be considered as a valid.
        smooth_fraction: float, 0.2 default
            It's the proportion of frequencies used in the discrete 
            Fourier Transform to smooth the curve. A lower value of
            smooth_fraction will result in a smoother curve. 
        periods: int, 24 default
            List of seasonal periods of the timeseries. 
            Multiple periods are allowed.
            Each period must be an integer reater than 0.
        treshold: float, 0.125 default
            Maximum distance used
        Window: integer (odd number) default 7
            Size of the window used for filtering with the Savitzky-Golay 
        show: boolean, False, default
            This inform if you want to plot the series with the starting 
            and ending of each cycle detected.
        iterations: int, 3 default
            Number of iterations used on LOWESS decomposition. 
            Iterations must be in the range (0,6].
            
    Returns
    -------
    pandas.dataframe:
        dataframe with the phenometrics
    timeseries plot with the start and end points
    """
    # Use decompose smoother to 
    import pandas
    from scipy.signal import savgol_filter
    from tsmoothie.smoother import DecomposeSmoother


    smoother = DecomposeSmoother(smooth_type='lowess', periods=periods,
                                 smooth_fraction=smooth_fraction, iterations=iterations)
    
    
    ts = savgol_filter(smoother.smooth(time_series).smooth_data[0], window, 3)
    
    if numpy.min(ts)<0:
        diff = time_series[0]-ts[0]
        ts = ts+diff
    diff=None
    
    #This function perform the filtering with the savitky-golay mehtod
    y = get_filtered_series(ts,window,treshold)
    
    #This function detect peaks on the timesries
    peaks = get_peaks(y,minimum_up) 
    #try:
        #This functions detect cycles and compute phenometrics from each of them
    dfp = domain(y,peaks,minimum_up,min_height )
    
    #clean dataframe
    indexNames = dfp[ (dfp['Ampl.'] < 0.01) ].index
    dfp.drop(indexNames , inplace=True)
    #except:
    #    return None
    
    #This show the plot with the timeseries
    if show == True:
        fig, ax = plt.subplots(figsize=(15, 5))
        plt.plot(time_series,label='TimeSeries')
        plt.plot(y,label='SG-Phenometrics')
        ax.scatter(dfp['Start'],dfp['Start val.'],label='Start point',color='green', s = 40)
        ax.scatter(dfp['End'],dfp['End val.'],label='End point',color='red', s = 40)
        #ax.scatter(peaks,y[peaks], label='Detected Inflection Points',color='black', s = 40)
        legend = ax.legend(loc='upper left', shadow=True, fontsize='large',ncol=5,bbox_to_anchor=(0,1.15))
        plt.ylim([0, 1])
        plt.show()
    
    return dfp