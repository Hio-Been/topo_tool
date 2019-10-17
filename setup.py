from setuptools import setup, find_packages

setup(
    name                = 'topo_tool',
    version             = '0.1',
    description         = 'topo_tool',
    author              = 'jeelab',
    author_email        = 'hiobeen@kist.re.kr',
    url                 = 'https://github.com/Hio-Been/topo_tool',
    download_url        = 'https://github.com/Hio-Been/topo_tool',
    install_requires    =  [],
    packages            = find_packages(exclude = []),
    keywords            = ['topo_tool'],
    python_requires     = '>=3',
    package_data        = {},
    zip_safe            = False,
    classifiers         = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
