from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("submatch")
except PackageNotFoundError:
    __version__ = "unknown"
