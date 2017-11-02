import os
import gzip

FILEPATH = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'ceo/data/blns.txt.gz')
"""Path to the file"""


def naughty_strings(filepath=FILEPATH):
    """Get the list of naughty_strings.

    By default this will get the strings from the blns.txt file

    Code is a simple port of what is already in the /scripts directory

    :param filepath: Optional filepath the the blns.txt file
    :returns: The list of naughty strings
    """

    strings = []
    with gzip.open(filepath, 'r') as f:

        # put all lines in the file into a Python list
        for string in f.read().split("\n"):
            if len(string) > 0:
                if (string[0] != "#" and "script" not in string 
                    and "SCRIPT" not in string 
                    and "alert" not in string and string[0] != '>'):
                    strings.append(string)
    return strings
