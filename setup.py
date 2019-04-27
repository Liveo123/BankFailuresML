import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bank_failure",
    version="0.0.1",
    author="CS6242",
    description="A package which implements random tree forrest to predict bank failure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.gatech.edu/amampilly3/CSE6242--project-repo",
    packages=setuptools.find_packages(),
    py_modules=['Analyze', 'CSV_Reader', 'interact'],
    install_requires=['numpy', 'pandas', 'sklearn', 'matplotlib', 'google-api-python-client', 
    'oauth2client', 'httplib2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)