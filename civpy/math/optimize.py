import scipy.optimize

__all__ = ['fsolve']


def fsolve(*args, **kwargs):
    """
    Finds the roots of a function. If the function fails to find a solution,
    an exception is raised. See :func:`scipy.optimize.fsolve` for list of
    parameters.
    """
    kwargs['full_output'] = True
    x, infodict, ier, mesg = scipy.optimize.fsolve(*args, **kwargs)

    if ier != 1:
        raise ValueError('{}\n{}'.format(mesg, infodict))

    return x
