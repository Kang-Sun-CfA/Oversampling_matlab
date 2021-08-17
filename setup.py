import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sci_level3",
    version="0.0.1",
    author="MSat Data Platform Team",
    author_email="msat@methanesat.org",
    description="In Progress MethaneSat Level2 to Level 3 Code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/methanesat-org/sci-level3",
    project_urls={
        "Bug Tracker": "https://github.com/orgs/methanesat-org/projects/1",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
)