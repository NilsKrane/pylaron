import numpy as np

def spectrum(J: np.ndarray, dE: float, range: float, E0=0., lor=0., gauss=0., Vmod=0., dQ=0) -> tuple[np.ndarray, np.ndarray]:
    ''' Calculate Franck-Condon spectrum for givin phonon density `J`. All input energies are considered in eV
    :param J:       1d-array of phonon density. Must start at zero energy and have energy resolution dE.
    :param dE:      Energy resolution
    :param range:   Energy range of spectrum (from `-range` to `range-dE`)
    :param E0:      Energy offset of spectrum
    :param lor:     HWHM of lorentzian broadening
    :param gauss:   HWHM of gaussian broadening
    :param Vmod:    Amplitude of Lock-in modulation
    :param dQ:      Change of charge state to determine energy direction of phonon progression. If `0` (default) sign of `E0` is used.

    :return: spectrum_x, spectrum_y
    '''

    hbar = 6.582119569e-16 # eV s
    dE=abs(dE)
    range=abs(range)
    x = np.arange(-range,range,dE)
    npnts = len(x)

    # x values in time-domain
    ft = np.fft.fftfreq(npnts,d=dE/(2*np.pi*hbar))

    # define "high energy side" of spectrum
    if dQ == 0:
        direction = np.sign(E0)
    else:
        direction = np.sign(-dQ)

    # transform J to time.-domain
    if direction >= 0:
        fy = np.fft.ifft(J*dE, n=npnts, norm='forward')
    else:
        fy = np.fft.fft(J*dE, n=npnts, norm='backward')
    corrfunc = np.exp(1j*E0/hbar*ft+fy) # "convolution" in time-domain

    # apply broadenings
    if lor:
        lorentzian_shape_fft = np.fft.ifft(lorentzian(x,lor)*dE,n=npnts, norm='forward')
        corrfunc *=lorentzian_shape_fft*np.exp(1j*x[0]/hbar*ft)
    
    if gauss:
        gaussian_shape_fft = np.fft.ifft(gaussian(x,gauss)*dE,n=npnts, norm='forward')
        corrfunc *=gaussian_shape_fft*np.exp(1j*x[0]/hbar*ft)

    if Vmod:
        lockin_shape_fft = np.fft.ifft(lockin(x,Vmod)*dE,n=npnts, norm='forward')
        corrfunc *=lockin_shape_fft*np.exp(1j*x[0]/hbar*ft)

    # transfrom back into energy-domain    
    Sy = np.fft.fft(corrfunc/dE, norm='forward')
    Sy *= 1/np.exp(np.sum(J*dE)) # renorm spectrum, such that np.sum(Sy)*dE = 1
    
    # x values in energy-domain
    Sx = np.fft.fftfreq(npnts)*dE*npnts

    return np.fft.fftshift(Sx), np.fft.fftshift(Sy).real

# ------------------------

def Jrect(x: np.ndarray, Er: float, xmin: float, xmax: float) -> np.ndarray:
    ''' Returns rectangular phonon dispersion, from `xmin` to `xmax` with total reorganization energy `Er`
    :param x:       1d-array of energy values
    :param Er:      Total reorganization energy
    :param xmin:    Lowest energy phonon mode
    :param xmax:    Highest energy phonon mode

    :return:        1d-array with length of `x`
    '''
    eta = 2*Er/(xmax**2 - xmin**2)
    return np.heaviside(x-xmin,1)*np.heaviside(xmax-x,1)*eta

def lorentzian(x: np.ndarray, hwhm: float) -> np.ndarray:
    ''' Lorentzian distribution
    :param x:       1d-array of energy values
    :param hwhm:    HWHM of lorentzian distribution

    :return:        1d-array with length of `x`
    '''
    return hwhm/np.pi/(hwhm**2+x**2)

def gaussian(x,hwhm):
    ''' Gaussian normal distribution
    :param x:       1d-array of energy values
    :param hwhm:    HWHM of gaussian distribution

    :return:        1d-array with length of `x`
    '''
    return np.sqrt(np.log(2)/np.pi)/hwhm*np.exp(-(np.log(2)*x**2/hwhm**2))

def lockin(x,Vmod):
    ''' Broadening function due to sinusoidal energy modulation (e.g. Lock-in amplifier)
    :param x:       1d-array of energy values
    :param Vmod:    Amplitude of sinusoidal modulation

    :return:        1d-array with length of `x`
    '''
    lockin_shape  = 2*np.sqrt(abs(Vmod**2-x**2))/np.pi/Vmod**2
    return lockin_shape * np.heaviside(Vmod+x,1)*np.heaviside(Vmod-x,1)
