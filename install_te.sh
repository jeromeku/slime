#!/bin/bash
# Ensure python dev headers are installed
# CUDA_HOME / CUDACXX set
current_file=`realpath $0`
current_dir=`dirname ${current_file}`
VENV_NAME=".slime-env"

export CUDNN_ROOT=${current_dir}/${VENV_NAME}/lib/python3.12/site-packages/nvidia/cudnn
if [[ -z ${CUDNN_ROOT} ]]; then
    echo "CUDNN_ROOT not set"
    exit 1
fi

export CUDNN_PATH=${CUDNN_ROOT}
export CPATH=${CUDNN_ROOT}/include:$CPATH
export LIBRARY_PATH=${CUDNN_ROOT}/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDNN_ROOT}/lib:$LD_LIBRARY_PATH
export CMAKE_CUDA_ARCHITECTURES=90a
export NVTE_CUDA_ARCHS=90a
export NVTE_CMAKE_EXTRA_ARGS="-DCMAKE_VERBOSE_MAKEFILE=1 -DCMAKE_EXPORT_COMPILE_COMMANDS=1"
export NVTE_FRAMEWORK=pytorch

EDITABLE_INSTALL=0
PATH_TO_TE=thirdparty/transformerengine
PACKAGE_PATH="transformer_engine[pytorch]"

[[ ${EDITABLE_INSTALL} -eq 1 ]] && PACKAGE_PATH="--editable ${PATH_TO_TE}" && rm -rf ${PATH_TO_TE}/build

CMD="uv pip install --no-build-isolation ${PACKAGE_PATH} -v  2>&1 | tee _te_install.log"

echo ">> ${CMD}"

eval "${CMD}"