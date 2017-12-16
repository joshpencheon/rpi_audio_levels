#!/usr/bin/env python

import math
import numpy as np
import pyaudio
import struct
from threading import Event, Lock, Thread
import time
import unicornhathd

FORMAT       = pyaudio.paInt16
RATE         = 44100
nFFT         = 2**8
nFRAMES      = 20
nTRACES      = 16
TRACE_HEIGHT = 16
RENDER_FPS   = 60

# Debugging: get some not-too-pretty output:
np.set_printoptions(precision=3, suppress=True, linewidth=2000)

class FrameBuffer(object):
    """
    Buffers a number of frames in a fixed-size two-dimensional
    array data structure. Provides synchronised access to buffer
    history, in reverse chronogical order.
    """
    def __init__(self, length, width):
        self.length = length
        self.width  = width
        self.index  = 0
        self.lock   = Lock()

        self.wipe()

    def wipe(self):
        with self.lock:
            self.frames = np.zeros(shape=(self.length, self.width))

    def push_frame(self, frame):
        with self.lock:
            self.index = (self.index - 1) % self.length
            self.frames[self.index] = frame

    def get_current_frame(self):
        return self.get_frames(1)

    def get_frames(self, limit=None):
        if limit is None: limit = self.length

        with self.lock:
            limit = min(limit, self.length)
            start = self.index
            stop  = start + limit

            slice = self.frames[start:stop]

            if stop > self.length:
                rest  = self.frames[0:(stop % self.length)]
                slice = np.concatenate((slice, rest))

            return slice

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
def to_level(bar, x_min, x_med, x_max):
    # There is no extent defined; return an "off" row:
    off = np.zeros_like(bar) - 1

    a = np.log(bar   + 1) - np.log(x_min + 1)
    b = np.log(x_max + 1) - np.log(x_min + 1)

    # Divide, unless the extent was zero-width:
    level = np.divide(TRACE_HEIGHT * a, b, out=off, where=b!=0)

    return np.clip(level, 0, TRACE_HEIGHT) - 1

# PyAudio callback, used to process data buffered from the microphone:
def callback(data, frame_count, time_info, flag):
    # Unpack expected number of floats, and then run through FFTs.
    # Due to symmetry, we can discared half of the results.
    signal = np.array(struct.unpack("%dh" % frame_count, data))
    ffts   = abs(np.fft.fft(signal, nFFT)[:nFFT/2])

    # Match FFTs with the static Bark values, and distribute
    # across the available space to get a complete frame:
    bark_levels = np.array([SCALED_BARKS, ffts]).T
    frame       = traces(bark_levels)

    frame_buffer.push_frame(frame)

    return (data, pyaudio.paContinue)

def render_loop(frame_buffer, start_event, stop_event):
    unicornhathd.rotation(-90)

    render_warmup(frame_buffer)
    start_event.set()
    while not stop_event.is_set(): render_frame(frame_buffer)
    render_warmdown(frame_buffer)

    unicornhathd.off()

def render_warmup(buf):
    step = 2 * np.pi / buf.length
    for i in range(0, buf.length) + range(buf.length, 0, -1):
        offset = step * i
        x = np.linspace(offset - np.pi, offset, buf.width)
        buf.push_frame(np.clip(np.sin(x), 0, None))
        render_frame(buf, render_max=False)
    buf.wipe()

def render_warmdown(buf):
    for i in range(0, buf.length):
        buf.push_frame(np.zeros(buf.width))
        render_frame(buf)

def render_frame(*args, **kwargs):
    target  = 1.0 / RENDER_FPS
    started = time.time()

    __render(*args, **kwargs)

    elapsed   = time.time() - started
    remaining = max(0, target - elapsed)
    time.sleep(remaining)

def __render(buf, render_max=True):
    frames    = buf.get_frames()
    age_limit = frames.shape[0]
    decay_exp = 4

    # Get the extent of all the datapoints in the frames:
    min_bar = np.amin(frames,   axis=0)
    med_bar = np.median(frames, axis=0)
    max_bar = np.amax(frames,   axis=0)

    # Map the most recent frame to a set of levels:
    levels = to_level(frames[0], min_bar, med_bar, max_bar)

    # Also, find the greatest bars across the frames.
    # Then weight based on index (older => dimmer, falling).
    max_bars   = np.amax(frames, axis=0)
    max_ages   = np.argmax(frames, axis=0)
    max_levels = to_level(max_bars, min_bar, med_bar, max_bar)

    # Linear (to [0, 1]), then exponential, decay:
    max_weights = np.argmax(frames, axis=0) * 1.0 / age_limit
    max_weights = (max_weights) ** decay_exp

    # Decay levels in height and brightness:
    max_levels      = max_levels - age_limit * max_weights
    max_intensities = 1 - decay_exp * max_weights

    unicornhathd.clear()

    for x, level in enumerate(levels):
        if render_max: turn_on(x, max_levels[x], v=max_intensities[x])
        # Draw current levels over the top of any decaying max levels...
        for y in range(0, int(level)): turn_on(x, y)

    unicornhathd.show()

def turn_on(x, y, v=1.00):
    if y < 0: return None
    unicornhathd.set_pixel_hsv(int(x), int(y), *hsv_for(x, y, v))

def hsv_for(x, y, v):
    fraction = (y + 1.0) / TRACE_HEIGHT
    if   fraction >= 0.85: return [0.00, 1.00, v]
    elif fraction >= 0.65: return [0.17, 1.00, v]
    else:                  return [0.27, 1.00, v]

def main():
    global FREQUENCY_BANDS, SCALED_BARKS, frame_buffer

    # FFTs will provide output in equally-wide bands spanning from
    # zero to half of the sample rate. The number of bands is half
    # the number of total FFTs. Each frequency band is then categorised
    # by where it would appear on the Bark Scale, which attempts to
    # account for differing sensitivies in human hearing across the
    # perceivable range of frequencies.
    FREQUENCY_BANDS = np.array(1.0 * np.arange(0, nFFT / 2) / nFFT * RATE)
    SCALED_BARKS    = np.array([scaled_bark(x) for x in FREQUENCY_BANDS])

    frame_buffer = FrameBuffer(nFRAMES, nTRACES)

    start_rendering = Event()
    stop_rendering  = Event()
    Thread(
        target=render_loop,
        args=(frame_buffer,start_rendering,stop_rendering,)
    ).start()

    while not start_rendering.is_set(): time.sleep(0.1)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
            channels=1,
            rate=RATE,
            input=True,
            stream_callback=callback)

    while stream.is_active():
        print "Press <ctrl-c> to stop..."

        while True:
            try:
                time.sleep(.5)
            except KeyboardInterrupt:
                stream.stop_stream()
                stop_rendering.set()
                break

    stream.close()
    p.terminate()

if __name__ == '__main__':
    main()
