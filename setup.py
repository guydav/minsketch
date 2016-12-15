from setuptools import setup

# TODO: Figure out what's the correct way to handle C-dependencies and RTD.
# TODO: See https://github.com/rtfd/readthedocs.org/issues/2549

setup(name='minsketch',
      version='0.1',
      description='A flexible implementation of several min-sketch variants',
      url='https://github.com/guydav/minsketch',
      author='Guy Davidson',
      author_email='guy@minerva.kgi.edu',
      license='MIT',
      packages=['minsketch'],
      zip_safe=True,)
