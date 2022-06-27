#!/bin/bash
#
# Install virtual environment that uses custom CT2 binary

set -e

CT2_ROOT_DIR=${PWD}
PYTHON_EXE=python3
VENV_PATH=venv
CT2_INSTALL_DIR=${CT2_ROOT_DIR}/install

${PYTHON_EXE} -m venv ${VENV_PATH}
${VENV_PATH}/bin/python -m pip install -U pip

# Need to set before `setup.py`, guided by [1]
export CTRANSLATE2_ROOT=${CT2_INSTALL_DIR}

${VENV_PATH}/bin/pip install -r ${CT2_ROOT_DIR}/python/install_requirements.txt
${VENV_PATH}/bin/pip install -e ${CT2_ROOT_DIR}/python

# Need for running python application correctly, guided by [1]
echo "export LD_LIBRARY_PATH=$(realpath --no-symlinks ${CT2_INSTALL_DIR}/lib)" >> ${VENV_PATH}/bin/activate

# References
# [1] https://github.com/OpenNMT/CTranslate2/blob/6c8faa7979/docs/installation.md#compile-the-python-wrapper
