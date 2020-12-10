import os, sys
import os.path

from setuptools import setup, find_packages

root = os.path.abspath(os.path.dirname(__file__))
package_name = "tpubar"
packages = find_packages(
    include=[package_name, "{}.*".format(package_name)]
)

_locals = {}
with open(os.path.join(package_name, "_version.py")) as fp:
    exec(fp.read(), None, _locals)

version = _locals["__version__"]
binary_names = _locals["binary_names"]

with open(os.path.join(root, 'README.md'), 'rb') as readme:
    long_description = readme.read().decode('utf-8')

setup(
    name=package_name,
    version=version,
    description="tpubar",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Tri Songz',
    author_email='ts@scontentenginex.com',
    keywords=['tpu', 'progress bar', 'monitoring', 'google cloud', 'tensorflow'],
    url='http://github.com/trisongz/tpubar',
    python_requires='>3.6',
    install_requires=[
        "tqdm>=4.50.0",
        "google-cloud-monitoring",
        "tensorflow",
        "psutil",
        "click",
        "pysimdjson",
        "tpunicorn",
    ],
    packages=packages,
    entry_points={
        "console_scripts": [
            "{} = {}.cli:cli".format(binary_name, package_name)
            for binary_name in binary_names
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
    ],
)