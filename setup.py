from setuptools import setup,find_packages
from torchhelper import __version__
setup(
    name='torchhelper',
    # 主版本，次版本，修订号？，bug修订号，...待定
    version=__version__,
    description='A torch helper for control log、train process（train/test/eval/checkpoint/save）',
    url='https://github.com/sailist/TorchHelper',
    author='sailist',
    author_email='sailist@outlook.com',
    license='Apache License 2.0',
    include_package_data = True,
    install_requires = [
      "torch","fire","tensorboard"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='torchhelper',
    packages=find_packages(),
    entry_points={
        'console_scripts':[
            # 'TODO = packagepath.pythonfilename:main'
        ]
      },
)