# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:00:09 2024

@author: uceerjp
"""

import pylsl

streams = pylsl.resolve_streams()

print("\n\n\n")

if len(streams):

    for i in range(len(streams)):

        print(str(i) + ": " + str(streams[i].name()))

    val = int(input("Please select stream>>"))

    if val in range(len(streams)):

        instream = pylsl.StreamInlet(streams[val])

        while True:

            chunk, time_stamp = instream.pull_sample()

            if len(chunk):
                print(chunk)


#%%



