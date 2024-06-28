from setuptools import setup

install_requires = ['pip','numpy','matplotlib','tensorflow']
extras_requires = ['jupyter']

setup(
    name='cgnn',
    version='0.1',
    description='Train conditional variational autoencoders to invert data',
    url='https://bitbucket.org/wmcalile/cgnn.git/',
    author='Andy McAliley',
    author_email='wmcalile@mymail.mines.edu',
    packages=['cgnn',],
    license='MIT license',
    install_requires = install_requires,
    extras_requires = extras_requires,
    long_description=open('README.md').read(),
    zip_safe=False
)
