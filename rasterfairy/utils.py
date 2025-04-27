def cmp_to_key(mycmp):
    """
    Convert `sorted` function from python2 to python3.

    This function is used to convert `cmp` parameter of python2 sorted
    function into `key` parameter of python3 sorted function.

    This code is taken from here:
    https://docs.python.org/2/howto/sorting.html#the-old-way-using-the-cmp-parameter

    :param mycmp: compare function that compares 2 values
    :return: key class that compares 2 values
    """
    "Convert a cmp= function into a key= function"
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

