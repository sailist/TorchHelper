from setuptools import setup,find_packages

setup(
    name='torchhelper',
    version='0.0.2.dev1',
    description='A torch helper for control log、train process（train/test/eval/checkpoint/save）',
    url='https://github.com/sailist/TorchHelper',
    author='sailist',
    author_email='sailist@outlook.com',
    license='Apache License 2.0',
    include_package_data = True,
    install_requires = [
      "torch",
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache License 2.0',
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