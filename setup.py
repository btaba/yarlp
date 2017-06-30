from distutils.core import setup
from setuptools import find_packages


def setup_package():
    config = {
        'name': 'yarlp',
        'version': '0.0.1',
        'description': 'yarlp',
        'author': 'Baruch Tabanpour',
        'author_email': '',
        'url': '',
        'install_requires': [
            'Click',
        ],
        'entry_points': '''
            [console_scripts]
            run_yarlp_experiment=yarlp.experiment.experiment:run_experiment
        ''',
        'packages': find_packages(exclude=["*.tests", "*.tests.*",
                                  "tests.*", "tests", "examples"])
    }

    setup(**config)


if __name__ == '__main__':
    setup_package()
