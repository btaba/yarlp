from distutils.core import setup
from setuptools import find_packages


def setup_package():
    config = {
        'name': 'yarlp',
        'version': '0.0.3',
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
            upload_to_openai=yarlp.experiment.experiment:upload_to_openai
            run_benchmark=yarlp.experiment.experiment:run_benchmark
            compare_benchmark=yarlp.experiment.experiment:compare_benchmark
            make_plots=yarlp.experiment.experiment:make_plots
        ''',
        'install_requires': [
            'gym[mujoco,atari,classic_control]',
            'baselines'
        ],
        'packages': find_packages()
    }

    setup(**config)


if __name__ == '__main__':
    setup_package()
