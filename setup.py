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
            continue_experiment=yarlp.experiment.experiment:continue_experiment
            upload_to_openai=yarlp.experiment.experiment:upload_to_openai
        ''',
        'packages': find_packages(exclude=["*.tests", "*.tests.*",
                                  "tests.*", "tests", "examples"])
    }

    setup(**config)


if __name__ == '__main__':
    setup_package()
