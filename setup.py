from setuptools import setup

extras = {
    'mujoco': [],
}

# dependency
all_dependencies = []
for group_name in extras:
    all_dependencies += extras[group_name]
extras['all'] = all_dependencies

setup(name='gym_multirotor',
      version='0.0.1',
      url='https://adipandas.github.io',
      author='Aditya M. Deshpande',
      author_email='adityadeshpande2010@gmail.com',
      install_requires=[
          'matplotlib',
          'scipy',
          'numpy',
      ],
      packages=['gym_multirotor', ],
      extras_require=extras,
      )
