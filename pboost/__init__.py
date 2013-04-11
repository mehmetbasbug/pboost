__version_info__ = {
    'major': 0,
    'minor': 1,
    'build': 6,
}


def get_version():
    """
    Return the formatted version information
    """
    vers = "%(major)i.%(minor)i.%(build)i" % __version_info__
    return vers

__version__ = get_version()
