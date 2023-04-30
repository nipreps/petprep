from setuptools import find_packages, setup

setup(
    entry_points={
        'console_scripts': [
            'petprep = petprep.__main__:main'
        ]
    },
    name='petprep',
    packages=find_packages(),
)
