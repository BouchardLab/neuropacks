from setuptools import setup, find_packages

setup(
    name='neuropacks',
    version='0.1',
    # What does your project relate to?
    keywords='neuroscience',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'h5py',
        'matplotlib',
        'numpy',
        'pyuoi',
        'scikit-learn',
        'scipy'
    ]
)
