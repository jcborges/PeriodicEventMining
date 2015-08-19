"""MaAndHellerstein: Provides an implementation for finding periodic behavior and discovering
unknown periods based on the paper of Ma and Hellerstein
called Mining Partially Periodic Event Patterns With Unknown Periods (2001)."""

__author__ = 'Julio De Melo Borges'
__email__ = "borges@teco.edu"

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import warnings


def _test_ctaus(ctaudict, delta):
    for key in ctaudict:
        ctau = ctaudict[key]
        p = int(np.mean(ctau))
        width = int(max(ctau)-min(ctau))
        if width > 2*delta:
            warning = "For p=%s the width of the inter arrival sequence is higher than 2*delta: %s. " \
                      "Correct delta parameter!" % (p, width)
            warnings.warn(warning)


def _extract_ctaus(labels, nexts):
    ctaudict = dict()
    for i in range(0, len(labels)):
        if labels[i] == -1:  #Noise or On-Segment Delimiters
            continue
        try:
            ctaudict[labels[i]].append(nexts[i])
        except KeyError:
            ctaudict[labels[i]] = [nexts[i]]
    return ctaudict


def _xsquared(ctau, delta, n, mean_arrival):
    period = np.mean(ctau)
    p_tau = 2 * delta * mean_arrival * np.exp(-1 * mean_arrival * period)
    thresh = np.sqrt(3.84 * n * p_tau * (1-p_tau)) + n * p_tau  #95% Confidence Level
    return thresh


def _extract_periods(ctaudict, delta, n, mean_arrival):
    periods = list()
    for key in ctaudict:
        ctau = ctaudict[key]
        thresh = _xsquared(ctau, delta, n, mean_arrival)
        if len(ctau) > thresh:
            periods.append(np.mean(ctau))
            print "p=%s, #Events=%s for Threshhold=%s" % (int(np.mean(ctau)), len(ctau), thresh)
    return periods


def run(point_sequence, delta):
    """
    This is the main and unique public method of this class.
    It runs the algorithm of Ma and Hellerstein for mining unknown periods in Partially Periodic Event Patterns
    :param point_sequence:  ordered collection of occurrence times of the target event as pandas time series index
                            (tseries.index.DatetimeIndex)
    :param delta: tolerance level in seconds
    :return: a list containing possible values for the period of the event
    """
    if not isinstance(point_sequence, pd.tseries.index.DatetimeIndex):
        raise TypeError, 'point_sequence is not of type tseries.index.DatetimeIndex, instead it is: '+str(type(point_sequence))
    n = len(point_sequence)
    mean_arrival = n/(max(point_sequence) - min(point_sequence)).total_seconds()
    nexts = [(point_sequence[i]-point_sequence[i-1]).total_seconds() for i in range(2, n)] #Inter-Arrival-Time

    #Calculate Ctau
    nexts_array = np.asarray(list(nexts))
    nexts_array.shape = (len(nexts), 1)
    db = DBSCAN(eps=delta, min_samples=2).fit(nexts_array)
    labels = db.labels_
    ctaudict = _extract_ctaus(labels, nexts)
    _test_ctaus(ctaudict, delta)

    #Statistical Tests for Ctau --> Discovering unkown periods
    return _extract_periods(ctaudict, delta, n, mean_arrival)
