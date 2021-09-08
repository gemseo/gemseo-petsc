'''
 * Copyright (c) {2021} {IRT-AESE}.
 * All rights reserved.
 *
 * Contributors:
 *    {INITIAL AUTHORS} - initial API and implementation and/or initial documentation
 *        @author: Francois Gallard
 *    {OTHER AUTHORS}   - {MACROSCOPIC CHANGES}
'''
from setuptools import setup, find_packages
setup(
    name='gemseo_petsc',
    version='0.0.1',
    packages=find_packages(where="src"),
    install_requires=[
        'gemseo',
        "petsc4py"],
    package_dir={"": "src"},
    include_package_data=True,
    author="Francois Gallard",
    author_email='contact@gemseo.org',
    license='LGPL3',
    url="www.irt-saintexupery.com"
)

