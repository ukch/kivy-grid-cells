"""
Although there's only three states set up here, there should be no hard limit
on the number of states available to the application - to have more than three
states, simply override these constants.
"""

class States(object):
    ILLEGAL = -1
    DEACTIVATED = 0
    FIRST = 1
    SECOND = 2


Colours = {
    States.ILLEGAL: (1, 0, 0, 1),  # Red
    States.DEACTIVATED: (0.5, 0.5, 0.5, 1),  # Grey
    States.FIRST: (1, 1, 1, 1),  # Black
    States.SECOND: (0, 0, 0, 1),  # White
}
