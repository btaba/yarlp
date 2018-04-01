from distutils.core import setup
from setuptools import find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


def setup_package():
    config = {
        'name': 'yarlp',
        'version': '0.0.9',
        'description': 'yarlp',
        'long_description': readme(),
        'author': 'Baruch Tabanpour',
        'author_email': 'baruch@tabanpour.info',
        'url': 'https://github.com/btaba/yarlp',
        'license': 'MIT',
        'install_requires': [
            'Click',
        ],
        'entry_points': '''
            [console_scripts]
            run_yarlp_experiment=yarlp.experiment.experiment:run_experiment
            upload_to_openai=yarlp.experiment.experiment:upload_to_openai
            compare_benchmark=yarlp.experiment.experiment:compare_benchmark
            make_plots=yarlp.experiment.experiment:make_plots
        ''',
        'install_requires': [
            'gym[mujoco,atari,classic_control]'
        ],
        'tests_require': ['nose'],
        'packages': find_packages(
            exclude=("tests", )),
        'keywords': [
            'reinforcement learning',
            'deep reinforcement learning',
            'experiment',
            'benchmark'
        ]
    }

    setup(**config)


if __name__ == '__main__':
    setup_package()
