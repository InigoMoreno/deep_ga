from setuptools import find_packages, setup
setup(
    name='deep_ga',
    packages=find_packages(include=['deep_ga']),
    version='0.1.0',
    description='Global Alignment using Deep Learning',
    install_requires=['numpy', 'plyfile', 'opencv-python'],
    author='Iñigo Moreno i Caireta',
    author_email='ignigomoreno@gmail.com',
    license='MIT',
    url='https://github.com/InigoMoreno/deep-ga'
)
