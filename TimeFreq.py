#Import libraries
import scipy.io as spio
from scipy import fftpack
import matplotlib.pyplot as plt
import numpy as np


#Process the dataset into samples
def process_positions(dataset, positions):
    output_range = 10
    classification_input = []
    for position in positions:
        lower = position - output_range
        upper = position + output_range
        classification_input.append(list(dataset[lower:upper]))
    return classification_input


#Put peak through fft
def process_FFT(time_sample):
    X = fftpack.fft(time_sample)
    return X


#Put all peaks through fft and put them in a list.
def process_all_FFT(time_samples):
    freq_samples = []
    for sample in time_samples:
        freq_samples.append(process_FFT(sample))
    unsorted_x = []
    #For all the samples, convert imaginary values into real values.
    for sample in freq_samples:
        new_sample = []
        for item in sample:
            new_sample.append(item.real)
            new_sample.append(item.imag)
        unsorted_x.append(list(new_sample))
    return unsorted_x



#Convert the dataset into frequency series samples.
def time_freq(dataset, positions):
    time_samples = process_positions(dataset, positions)
    freq_samples = process_all_FFT(time_samples)
    return freq_samples

