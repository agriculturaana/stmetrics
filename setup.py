import setuptools

with open('README.rst', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='stmetrics',
    version='0.1.8',
    author='Brazil Data Cube Team',
    author_email='brazildatacube@dpi.inpe.br',
    description='A package to compute features from Satellite Image Time Series (SITS).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/brazil-data-cube/stmetrics/',
    packages=['stmetrics'],
    install_requires=[
        'affine==2.4.0',
        'attrs==25.3.0',
        'beautifulsoup4==4.13.4',
        'certifi==2025.4.26',
        'charset-normalizer==3.4.2',
        'click==8.2.1',
        'click-plugins==1.1.1',
        'cligj==0.7.2',
        'connected-components-3d==3.23.0',
        'contourpy==1.3.2',
        'cycler==0.12.1',
        'descartes==1.1.0',
        'dtaidistance==2.3.13',
        'fastremap==1.16.1',
        'fiona==1.10.1',
        'fonttools==4.58.2',
        'future==1.0.0',
        'geopandas==1.1.0',
        'idna==3.10',
        'joblib==1.5.1',
        'kiwisolver==1.4.8',
        'libpysal==4.13.0',
        'llvmlite==0.44.0',
        'matplotlib==3.10.3',
        'nolds==0.6.2',
        'numba==0.61.2',
        'numpy==2.2.6',
        'packaging==25.0',
        'pandas==2.3.0',
        'pillow==11.2.1',
        'platformdirs==4.3.8',
        'pointpats==2.5.1',
        'pyogrio==0.11.0',
        'pyparsing==3.2.3',
        'pyproj==3.7.1',
        'python-dateutil==2.9.0.post0',
        'pytz==2025.2',
        'rasterio==1.4.3',
        'rasterstats==0.20.0',
        'requests==2.32.4',
        'scikit-learn==1.7.0',
        'scipy==1.15.3',
        'setuptools==80.9.0',
        'shapely==2.1.1',
        'simdkalman==1.0.4',
        'simplejson==3.20.1',
        'six==1.17.0',
        'soupsieve==2.7',
        'threadpoolctl==3.6.0',
        'tqdm==4.67.1',
        'tsmoothie==1.0.5',
        'typing-extensions==4.14.0',
        'tzdata==2025.2',
        'urllib3==2.4.0',
        'xarray==2025.6.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
    ],
)
