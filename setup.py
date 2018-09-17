from setuptools import setup, find_packages

setup(
    name='self_organizing_map',
    version='0.1',
    packages=find_packages(),
    url='',
    license='MIT',
    author='Kevin Rieck',
    author_email='kevin.rieck@fau.de',
    description='Self organizing map for process monitoring',
    install_requires=[
        'tensorflow>=1', 'numpy', 'matplotlib', 'pandas', 'scikit-learn', 'scipy', 'scikit-image'
    ]
)
