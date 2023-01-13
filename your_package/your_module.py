"""Top level description of your module."""


class TemplateClass(object):
    """
    The template class does nothing.

    See https://numpydoc.readthedocs.io/en/latest/format.html.
    For how this docstring is layed out.

    Attributes
    ----------
    foo : int
        A useless attribute.
    bar : int
        Another useless attribute.

    Parameters
    ----------
    foo : float, optional
        The first number to multiply by, default is 0
    bar : float, optional
        The second number to multiply by, default is 0

    Methods
    -------
    __add__(self, other)
        return (self.foo * self.bar) + (other.foo * other.bar)

    """

    def __init__(self, foo=0, bar=0):
        """See your_package.your_module.TemplateClass."""
        self.bar = bar
        self.foo = foo

    def __add__(self, other):
        return (self.foo * self.bar) + (other.foo * other.bar)
