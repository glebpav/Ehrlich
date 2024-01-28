from setuptools import setup, find_packages


def readme():
  with open('README.md', 'r') as f:
    return f.read()


setup(
  name='ehrlich',
  version='0.0.1',
  author='X-Syna',
  packages=['ehrlich', 'ehrlich/utils'],
  author_email='example@gmail.com',
  description='This is the simplest module for quick work with files.',
  long_description=readme(),
  long_description_content_type='text/markdown',
  # packages=find_packages(),
  install_requires=['requests>=2.25.1'],
  classifiers=[
    ''
  ],
  keywords='',
  project_urls={
    'GitHub': 'your_github'
  },
  python_requires='>=3.6',
)