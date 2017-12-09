#!/usr/bin/env python

import math
import numpy as np
import pyaudio
import struct
import time
import unicornhathd

FORMAT       = pyaudio.paInt16
RATE         = 44100
nFFT         = 2**8
nFRAMES      = 5
nTRACES      = 16
TRACE_HEIGHT = 16

# Debugging: get some not-too-pretty output:
np.set_printoptions(precision=0, suppress=True, linewidth=2000)

# For the given frequency, return the position on the Bark scale:
def raw_bark(f):
    return 13 * math.atan(0.00076 * f) + 3.5 * math.atan((f / 7500)  ** 2)

# For the given frequency, return the position on the Bark scale,
# scaled linearly to fit into the number of traces available:
def scaled_bark(f):
    scaled = round(raw_bark(f) * nTRACES / raw_bark(RATE/2))
    return min(max(scaled, 0), nTRACES - 1)

# Distribute the given levels across the number of traces available:
def traces(bark_levels):
    traces = np.zeros(nTRACES)

    # Start by aggregating levels by their bark value:
    for (bark, level) in bark_levels: traces[int(bark)] += level

    # Now, fill empty buckets from below by distributing non-zero
    # buckets. This is most useful at lower frequency ranges, where
    # equal range bands of FFTs across the spectrum don't match up
    # with the increased sensitivity represented by the Bark scale.
    spread_to = nTRACES - 1
    for index in range(spread_to, -1, -1):
        # We're going to want to try and spread over this index: 
        if 0 == traces[index]: continue

        if spread_to != index:
            # got a value, and will spread it up over previous zeros:
            to_spread = traces[index] / (spread_to - index + 1)
            for target in range(index, spread_to + 1):
                traces[target] = to_spread

        # Reset the floor to below this index
        spread_to = index - 1

    return traces

# Scale the bar within the bounds of x:
def to_level(bar, x_min, x_max):
    grad = TRACE_HEIGHT / (x_max - x_min)
    return np.clip(grad * (bar - x_min), 0, TRACE_HEIGHT - 1)

# Globals used to track the state of the buffered audio:
i      = -1
frames = np.zeros(shape=(nFRAMES, nTRACES))

# PyAudio callback, used to process data buffered from the microphone:
def callback(data, frame_count, time_info, flag):
    global i, frames

    # Unpack expected number of floats, and then run through FFTs.
    # Due to symmetry, we can discared half of the results.
    signal = np.array(struct.unpack("%dh" % frame_count, data))
    ffts   = abs(np.fft.fft(signal, nFFT)[:nFFT/2])

    # Match FFTs with the static Bark values, and distribute
    # across the available space to get a complete frame:
    bark_levels = np.array([SCALED_BARKS, ffts]).T
    frame       = traces(bark_levels)

    # Store the frame in our circular history:
    i = (i + 1) % nFRAMES
    frames[i] = frame

    render()

    return (data, pyaudio.paContinue)

def render():
    # Get a set of bars which represents the greatest across
    # all frames, as well as absolute bounds across all values:
    max_bars = np.amax(frames, axis=0)
    max_bar  = np.amax(frames)
    min_bar  = np.amin(frames)

    levels     = to_level(frames[i], min_bar, max_bar)
    max_levels = to_level(max_bars, min_bar, max_bar)

    unicornhathd.clear()

    for x, level in enumerate(levels):
        for y in range(0, int(level)):
            unicornhathd.set_pixel(x, y, 255, 255, 255)

        unicornhathd.set_pixel(x, int(max_levels[x]), 255, 0, 0)

    unicornhathd.show()

def main():
    global FREQUENCY_BANDS, SCALED_BARKS

    # FFTs will provide output in equally-wide bands spanning from
    # zero to half of the sample rate. The number of bands is half
    # the number of total FFTs. Each frequency band is then categorised
    # by where it would appear on the Bark Scale, which attempts to
    # account for differing sensitivies in human hearing across the
    # perceivable range of frequencies.
    FREQUENCY_BANDS = np.array(1.0 * np.arange(0, nFFT / 2) / nFFT * RATE)
    SCALED_BARKS    = np.array([scaled_bark(x) for x in FREQUENCY_BANDS])

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
            channels=1,
            rate=RATE,
            input=True,
            stream_callback=callback)

    while stream.is_active():
        time.sleep(600)
        stream.stop_stream()

    stream.close()
    p.terminate()

    unicornhathd.off()

if __name__ == '__main__':
    main()
