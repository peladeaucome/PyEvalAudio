# PyEvalAudio

## Porting PQevalAudio to Python

This repo is a port of the PQevalAudio Matlab code to Python.

## How to
Only the numpy version was done. To use, just import peaq_numpy and create a ```peaq_numpy.PEAQ()``` instance.
This object then has ```compute_peaq``` and ```compute_2fmodel``` methods to compute either the PEAQ score (ODG) or the 2fmodel.