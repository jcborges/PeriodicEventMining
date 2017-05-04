"""Borges: Implementation of MDPEP Algorithm.
Published in: Dynamically Periodic Event Patterns in Industrial Log Files
The 3rd IEEE International Conference on Smart Data
SmartData 2017
"""

__author__ = 'Julio De Melo Borges'
__email__ = "borges@teco.edu"

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import warnings
import copy, sys

#R Interface
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
randtests = importr("randtests")  #Install this library if you dont have it


def is_random_process(inter_arrival_time, confidence_level=0.05):
    rt = randtests.difference_sign_test(np.array(inter_arrival_time))
    pvalue = np.array(rt[4])[0]
    if pvalue < confidence_level:
        print "P-Value:", pvalue
        return False
    print "There is not enough evidence for non-randomness. P-Value = %s" % pvalue
    return True


def _test_ctaus(ctaudict, delta):
    for key in ctaudict:
        ctau = ctaudict[key]
        p = int(np.mean(ctau))
        width = int(max(ctau)-min(ctau))
        if width > 2*delta:
            warning = "For p=%s the width of the inter arrival sequence is higher than 2*delta: %s. " \
                      "Correct delta parameter!" % (p, width)
            warnings.warn(warning)


def _filter_ctau(ctaudict, delta, minsup):
    new_ctau = copy.copy(ctaudict)
    for key in ctaudict:
        if key == -1:
            sys.exit("Invalid Key Number")
        ctau = ctaudict[key]
        p = np.mean(ctau)
        corrected_ctau = []
        for v in ctau:
            if (v <= p + delta) and (v >= p - delta):
                corrected_ctau.append(v)
        new_ctau.pop(key, None)
        if len(corrected_ctau) > minsup:
            new_ctau[key] = corrected_ctau
    return new_ctau


def _extract_ctaus(labels, nexts, delta, minsup):
    ctaudict = dict()
    for i in range(0, len(labels)):
        if labels[i] == -1:  #Noise or On-Segment Delimiters
            continue
        try:
            ctaudict[labels[i]].append(nexts[i])
        except KeyError:
            ctaudict[labels[i]] = [nexts[i]]
    return _filter_ctau(ctaudict, delta, minsup)


def _extract_periods(ctaudict):
    periods = list()
    for key in ctaudict:
        ctau = ctaudict[key]
        periods.append(np.mean(ctau))
        print "p=%s, #Events=%s" % (int(np.mean(ctau)), len(ctau))
    return periods


def _test_nsr(labels):
    sl = pd.Series(labels)
    try:
        nsr = float(len(sl[sl == -1]))/float(len(sl))
    except ZeroDivisionError:
        nsr = len(sl)
    if nsr > 0.5:
        warning = "Very high noise to signal ratio: %s. ABORTING!" % nsr
        warnings.warn(warning)
    print "NSR:",nsr
    return nsr > 0.5


def _is_in_segment(periods, delta, next_distance):
    for p in periods:
        if (next_distance >= p - delta) and (next_distance <= p + delta):
            return True
    return False


def run(point_sequence, delta, minsup):
    """
    This is the main and unique public method of this class.
    It runs the algorithm of Borges et al. for mining unknown periods in Partially Periodic Event Patterns
    :param point_sequence:  ordered collection of occurrence times of the target event as pandas time series index
                            (tseries.index.DatetimeIndex)
    :param delta: tolerance level in seconds
    :return: a list containing possible values for the period of the event
    """
    if not isinstance(point_sequence, pd.tseries.index.DatetimeIndex):
        raise TypeError, 'point_sequence is not of type tseries.index.DatetimeIndex, instead it is: '+str(type(point_sequence))
    n = len(point_sequence)
    mean_arrival = n/(max(point_sequence) - min(point_sequence)).total_seconds()
    nexts = [(point_sequence[i]-point_sequence[i-1]).total_seconds() for i in range(2, n)]  #Inter-Arrival-Time

    if is_random_process(nexts):
        return None

    #Calculate Ctau
    nexts_array = np.asarray(list(nexts))
    nexts_array.shape = (len(nexts), 1)
    db = DBSCAN(eps=delta, min_samples=minsup).fit(nexts_array)
    labels = db.labels_
    ctaudict = _extract_ctaus(labels, nexts, delta, minsup)
    abortsnr = _test_nsr(labels)
    if abortsnr:
        return None
    _test_ctaus(ctaudict, delta)
    return _extract_periods(ctaudict)


def get_segments(point_sequence, delta, periods):
    """
    Returns the On-Segments as a list of timestamps of the periodic event
    :param point_sequence:  ordered collection of occurrence times of the target event as pandas time series index
                            (tseries.index.DatetimeIndex)
    :param delta:           tolerance level in seconds
    :param periods:         A list of possible periods (results of run method)
    :return: a list containing list segments of type tseries.index.DatetimeIndex
    """
    segments = []
    segment = []
    for i in range(0,len(point_sequence)-1):
        next_distance = (point_sequence[i+1] - point_sequence[i]).total_seconds()
        segment.append(point_sequence[i])
        if not _is_in_segment(periods, delta, next_distance):
            if len(segment) > 1:
                segments.append(segment)
            segment = []
    return segments