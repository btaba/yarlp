from distutils.core import setup
from setuptools import find_packages


def setup_package():
    config = {
        'name': 'yarlp',
        'version': '0.0.0',
        'description': 'yarlp',
        'author': 'Baruch Tabanpour',
        'author_email': '',
        'url': '',
        'packages': find_packages(exclude=["*.tests", "*.tests.*",
                                  "tests.*", "tests", "examples"])
    }

    setup(**config)


if __name__ == '__main__':
    setup_package()
