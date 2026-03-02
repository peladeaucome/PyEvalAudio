import setuptools
import codecs
import os.path

with open("README.md", "r") as fh:
    long_description = fh.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="PyEvalAudio",  # Replace with your own username
    version=get_version("PyEvalAudio/__init__.py"),
    author="Peladeau",
    author_email="come.peladeau@telecom-paris.fr",
    description="A numpy implementation of peaq",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/peladeaucome/PyEvalAudio",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "scipy>=1.11.3",
        "numpy>=1.26.1",
        "numba>=0.58.1",
    ],
)