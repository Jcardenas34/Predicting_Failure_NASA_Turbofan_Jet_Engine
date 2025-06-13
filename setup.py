from setuptools import setup, find_packages

setup(
    name="predicting_failure",        # Change to your package's name
    version="0.0.1",
    package_dir={"": "src"},  
    packages=find_packages(),      # Automatically finds submodules
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tables",
        "fastapi",
        "uvicorn",
        "flask",
        "torch",

        # Add any runtime deps here (not dev tools or testing libs)
    ],
    author="Juan Cardenas",
    description="A Anomaly detection package that will determine RUL",
    python_requires=">=3.6",
)