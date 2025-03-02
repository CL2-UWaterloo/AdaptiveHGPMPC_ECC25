from setuptools import setup, find_packages

setup(
    name='adap_hgpmpc',  # Replace with your package name
    version='1.0',  # Replace with your version
    packages=find_packages(where='src'), # important: specify where the package is
    package_dir={'': 'src'}, #important: specify where the package is
    install_requires=[
        # List any dependencies here (e.g., 'numpy', 'requests')
    ],
)