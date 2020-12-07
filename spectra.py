from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.signal as sig
import numpy as np
import os


def _expand_to_range(inp):
    return (0, inp) if np.isscalar(inp) else inp


def _nearest_power_two(x):
    return int(pow(2, np.ceil(np.log2(x))))


def _LFM_angles(Fs, start_f, end_f, T):
    """
    Array of angles corresponding to linear chirp.
    :param Fs: Sampling frequency
    :param start_f: Start frequency, in Hz,.
    :param end_f: End frequency, in Hz
    :param T: Total time, in seconds

    TODO: Exponential chirp
    """
    t  = np.linspace(0, T, int(T*Fs))
    bw = end_f - start_f
    return 2.0*np.pi*start_f*t + np.pi*(bw/T)*t**2


def _plot_colour_map(x, y, vals, ax, cmap=None, color_bar=False):
    """
    Plot colour map, and add colour bar key
    """
    if cmap is None:
        cmap = plt.cm.bone

    x = _expand_to_range(x)
    y = _expand_to_range(y)

    # Plot
    im = ax.pcolorfast(x, y, vals, cmap=cmap)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.grid(False)

    # Add colour bar
    if color_bar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='1%', pad=0.05)
        ax_fig = cax.figure
        cbar = ax_fig.colorbar(im, cax=cax, ticks=[])
        cbar.ax.set_yticklabels([])
        cbar.set_ticks([])


def plot_spectrum(data, Fs, ax,
        norm_func=None,
        NFFT=None,
        skip=None,
        numzeropad=2,
        freqrange=None,
        cmap=None,
        win_func=np.blackman,
        color_bar=False,
        saturate_prec=1):
    """
    Plot a spectrogram image.

    Parameters
    ----------
    :param data: Numpy 1d-array timeseries amplitudes.
    :param Fs: Sampling rate in Hz
    :param fig: matplotlib figure object
    :param ax: matplotlib figure axis
    :param norm_func: Normalisation function
    :param NFFT: length of FFT window
    :param skip: Number of overlaps between windows, skip > 1
    :param numzeropad: Zeropadding length for fft
    :param freqrange: Only limit on plot axis.
    :param cmap: See http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
    :param win_func: windowing function applied to time slice for fft.
    :param color_bar: Add colour bar as key
    :param saturate_prec: Saturate pixels to this percentile for better contrast
    """
    # Default plot params based on sampling frequency
    if NFFT is None:
        NFFT = _nearest_power_two(Fs*10) # Default to 10s
    if skip is None:
        skip = max(1, NFFT // 2)

    # Generate spectrogram
    f, t, s = sig.spectrogram(data, nperseg=NFFT, nfft=numzeropad*NFFT,
                                fs=Fs, window=win_func(NFFT), noverlap=NFFT-skip)

    # Adjust image values
    if norm_func is not None:
        s = norm_func(s)

    if 0 < saturate_prec < 100:
        s = np.clip(s, np.percentile(s, saturate_prec),
                       np.percentile(s, 100-saturate_prec))
    else:
        print("WARNING: Saturation percentile ignored, must be between 0 & 100")

    # Plot result
    if freqrange is None:
        _plot_colour_map(t, f, s, ax, cmap, color_bar)
    else:
        freqrange = _expand_to_range(freqrange)
        sub_freq_idx = np.logical_and(freqrange[0] <= f, f <= freqrange[-1])
        _plot_colour_map(t, f[sub_freq_idx], s[sub_freq_idx, :], ax, cmap, color_bar)


def plot_spectrum_db(data, Fs, ax, **kwargs):
    """
    Plot a spectrogram, with units in dB.
    See plot_spectrum for param details.
    """
    to_dB = lambda x: 10 * np.log10(x)
    plot_spectrum(data, Fs, ax, norm_func=to_dB, **kwargs)


def triangle_wave(x, n_terms):
    """
    Creates an approx. triangle wave using the Fourier sequence:
    :param x: Array of angles (in radians)
    :param n_terms: Number of harmonic terms, including fundamental freq.

    TODO: Other waveforms
    """
    triangle = np.zeros(x.size)
    for i in range(n_terms):
        triangle += np.sin((2.0*i-1)*x) * ((-1)**i)/((2.0*i-1)**2.0)
    return triangle


def plot_as_image(signal, Fs, cmap, save_name=None, width=1584, height=396, dpi=130):
    """
    Plot spectrogram of given signal as an image, with no borders or ticks.
    :param signal: Signal to plot
    :param Fs: Sampling frequency (Hz)
    :param cmap: Matplotlib colour map
    :param save_name: Save location for image
    :param width: Image width
    :param height: Image height
    :param dpi: Image DPI
    """
    fig, ax = plt.subplots(1, 1, figsize=(width/dpi, height/dpi), dpi=dpi)
    plot_spectrum_db(signal, Fs, ax, cmap=cmap)
    plt.tick_params(axis='both', which='both',
                    bottom=False, top=False,
                    labelbottom=False, right=False,
                    left=False, labelleft=False)
    plt.tight_layout(0.0)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if save_name is None:
        # No save name specified, put it in the script dir
        save_name = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "alias_image_out.png")
    fig.savefig(save_name, bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close(fig)


if __name__ == '__main__':
    # Construct signal
    Fs = 480    # Resolution in y direction

    # ### Full frequency plot ###
    # # Useful for exploration e.g. finding interesting places to zoom in on
    # cmap = "rainbow"    # Colour map, see https://matplotlib.org/tutorials/colors/colormaps.html
    # HARM_TERMS = 35     # Number of harmonic terms in waveform, more -> crazier patterns, but gets messy quickly
    # START_F = 0         # Start frequency of chirp (in Hz)
    # END_F   = Fs        # End frequency of chirp (in Hz), generally convenient to define these in terms of Fs
    # T = 10000           # Resolution in x direction, higher value brings out finer details but can actually make
    #                     # solid lines look less bold and more messy, so higher isn't always better
    #                     # Recommended to keep low during exploration, full resolution can be pretty slow
    # sig_to_plot = triangle_wave(_LFM_angles(Fs, START_F, END_F, T), HARM_TERMS)

    # ### Non-alias to alias ###
    # # Nice transition from clean lines to aliasing, Bladerunner vibes
    # cmap = "inferno_r"    # _r flips the colour map so that the lines are dark
    # HARM_TERMS = 43       # Enough for interesting patterns on the right, but not overwhelming
    # START_F = -Fs / 64    # Just a bit before Fs == 0
    # END_F   = Fs / 16     # Integer multiple of Fs so that the image ends on a "convergence" point
    # T = 100000
    # sig_to_plot = triangle_wave(_LFM_angles(Fs, START_F, END_F, T), HARM_TERMS)

    # ### Perspective Illusion ###
    # # Lots of harmonics here hide that this is a triangle wave at all and give an interesting corridor effect
    # cmap = "bone_r"     # Also works well with "pink", simple colours better in general
    # HARM_TERMS = 115
    # F_MARGIN = Fs / 32              # Small distance for margin on either side of features
    # START_F = (Fs / 4) - F_MARGIN   # At Fs/4, all harmonics converge at the center
    # END_F   = (Fs / 2) + F_MARGIN   # At Fs/2, all harmonics converge at the top
    # T = 20000                       # Smaller T actually brings out "skeleton" lines better
    # sig_to_plot = triangle_wave(_LFM_angles(Fs, START_F, END_F, T), HARM_TERMS)

    ### Hourglass ###
    # All harmonics converge at Fs/4, checking this was the whole reason I made this script
    cmap = "twilight_shifted_r"     # I really love this colour map, basically every version of "twilight" looks cool
    HARM_TERMS = 55
    F_CENTER = Fs / 4       # Convergence point
    F_HWIDTH = Fs / 128     # Small distance on either side of image center
    START_F = F_CENTER - F_HWIDTH
    END_F   = F_CENTER + F_HWIDTH
    T = 100000
    sig_to_plot = triangle_wave(_LFM_angles(Fs, START_F, END_F, T), HARM_TERMS)

    # Plot constructed signal
    plot_as_image(sig_to_plot, Fs, cmap, width=1584, height=396)
