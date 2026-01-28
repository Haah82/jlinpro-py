from setuptools import setup, find_packages

setup(
    name="pylinpro",
    version="0.1.0",
    description="Advanced Structural Analysis System - Python/Streamlit Migration of JLinPro",
    author="HST.AI Engineering",
    author_email="ha.nguyen@hydrostructai.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "matplotlib>=3.7.0",
        "deap>=1.4.0",
        "pydantic>=2.0.0",
        "pytest>=7.4.0",
        "black>=23.0.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
