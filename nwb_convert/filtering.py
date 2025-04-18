import scipy.signal as signal
import numpy as np

# -- frequency filtering functions
def _filter(b, a, x):
    """zero-phase digital filtering that handles nans"""
    nan_mask= np.isnan(x)
    is_nans = (nan_mask.sum() > 0)
    # temporarily replace nans with mean for filtering
    if is_nans:
        x[nan_mask] = np.nanmean(x)

    # apply filtering
    x = signal.filtfilt(b, a, x, axis=0)

    # put nans back where they were
    if is_nans:
        x[nan_mask] = np.nan

    return x

# notch filter
def apply_notch_filt(x, fs, notch_cent_freq, notch_bw_freq):
    """apply recursive notch filtering at specified frequencies"""
    for n_freq, n_bw in zip(notch_cent_freq, notch_bw_freq):
        # python iirnotch expects quality factor (Q)
        Q = n_freq/n_bw
        b, a = signal.iirnotch(n_freq, Q, fs=fs)
        x = _filter(b, a, x)
    return x

# high-pass filter
def apply_butter_filt(x, fs, filt_type, cutoff_freq, filt_order=4):
    """apply 4th order butterworth filtering at specified frequency"""
    # design filter
    b, a = signal.butter(filt_order, cutoff_freq, filt_type, fs=fs)

    # apply filter
    x = _filter(b,a,x)

    return x

# define savistky golay differentiation function
def apply_savgol_diff(x, window_length, polyorder, deriv, delta):
    """apply savitsky-golay differentiation"""

    y = signal.savgol_filter(x, window_length=window_length,
                             polyorder=polyorder,
                             deriv=deriv,
                             delta=delta,
                             mode='constant')
    return y

def apply_bayes_filter(x, fs, nbins, sigmax, alpha, beta, obs_model_type, pointmax):
    """runs bayesian filtering on emg data arrays

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz)
    nbins : int
        number of bins to create histogram
    sigmax : float
        maximum value of histogram
    alpha : float
        controls the time-varying diffusion process underlying the latent variable evol.
    beta : float
        controls the probability of larger jumps of the latent variable
    obs_model_type : str
        observation model type. Either 'Gauss' or 'Laplace' currently supported
    pointmax : bool
        whether to return maximum aposterior (MAP) est. if True, else return expectation value

    Returns
    -------
    np.array of bayesian filtered output

    Example
    -------

    """

    def time_evolve_prior(prior, dt, dsigma, alpha, beta):
        """Chapman Kolmogorov stochastic differential equation
        captures dynamics of EMG signal.
        """

        shifted_prior_1 = np.insert(prior[0:-1], 0, prior[0])
        shifted_prior_2 = np.append(prior[1:], prior[-1])
        augmented_prior = (shifted_prior_1 + (-2*prior) + shifted_prior_2)/np.power(dsigma,2)
        alpha_term = dt*alpha*augmented_prior
        beta_term = dt*beta + (1 - dt*beta)* prior
        timeevol_prior = alpha_term + beta_term

        return timeevol_prior

    def compute_likelihood(data, bins, obs_model_type):
        """compute likelihood from respective observation model given data."""

        def gauss_likelihood(data, bins):
            """gaussian likelihood"""
            numerator = np.exp(-0.5 *np.divide(np.power(data,2), np.power(bins, 2)))
            likelihood = np.divide(numerator, bins)

            return likelihood

        def laplace_likelihood(data, bins, obs_model_type):
            """laplacian likelihood"""
            numerator = np.exp(np.divide(-np.abs(data), bins))
            likelihood = np.divide(numerator, bins)

        # measurement/observation update
        if obs_model_type == 'Gauss':
            likelihood = gauss_likelihood(data,bins)
        elif obs_model_type == 'Laplace':
            likelihood = laplace_likelihood(data,bins)
        else:
            print('Unknown option for likelihood model')

        return likelihood
    def compute_posterior(likelihood, prior):
        """compute posterior distribution given likelihood and prior"""
        posterior = np.multiply(likelihood, prior)
        # normalize
        posterior = posterior/np.sum(posterior)

        return posterior
    def point_estimation(posterior, bins, pointmax):
        """compute point estimate used as model output"""
        # point estimation
        if pointmax:
            # maps: maximum aposteriori standard deviation
            index = np.argmax(posterior)
            output = bins[index]
        else:
            # expectation value
            output = sum(posterior * bins)

        return output

    # one order of magnitude larger than stddev
    #sigmax = np.std(x)*10

    # create right bin edges
    bins = np.linspace(0, sigmax, nbins+1)
    bins = np.delete(bins,0) # eliminate 0 bin

    # bin size
    dt = 1/fs
    # histogram bin width
    dsigma = bins[1] - bins[0]

    # init prior (uniform dist.)
    prior = np.ones(nbins)*(1/nbins)

    # create array to hold output
    bf_filt_x = np.empty(x.shape)
    samples = x.values.tolist()
    for i, x_i in tqdm(enumerate(samples)):
        # apply time evolution to prior, compute likelihood, posterior, point estimate
        timeevol_prior = time_evolve_prior(prior, dt, dsigma, alpha, beta)
        likelihood = compute_likelihood(x_i, bins, obs_model_type)
        posterior = compute_posterior(likelihood, timeevol_prior)
        bf_filt_x[i] = point_estimation(posterior, bins, pointmax)

        # set prior for next sample with posterior from this sample
        prior = posterior

    return bf_filt_x

# -- resample functions
def resample_column(x, target_fs, source_fs):
    """Apply order 500 Chebychev filter and then downsample"""
    resampled_x = signal.resample_poly(
        x, int(target_fs), int(source_fs))
    return resampled_x

# -- transform functions
# should likely go in a different file but only transform function for now
def rectify(x):
    """apply absolute value rectification"""
    return np.abs(x)

def mean_center(x):
    """ apply mean centering """
    mean_x = np.nanmean(x,axis=0)
    return x - mean_x
