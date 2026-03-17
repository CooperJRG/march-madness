import scipy.signal
try:
    import scipy.signal.signaltools
except AttributeError:
    pass
if not hasattr(scipy.signal, '_centered') and not hasattr(scipy.signal, 'signaltools'):
    import scipy.signal as sigt
else:
    sigt = scipy.signal.signaltools if hasattr(scipy.signal, 'signaltools') else scipy.signal
if not hasattr(sigt, '_centered'):
    def _centered(arr, newsize):
        return arr
    sigt._centered = _centered

import statsmodels.api as sm
print("Successfully imported statsmodels!")
