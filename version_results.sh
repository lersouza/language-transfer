#!/bin/bash

VERSION_TAG=$(date '+%Y-%m-%d')
TARGET_DIR="results/history-${VERSION_TAG}"

mkdir -p ${TARGET_DIR}

ls results/*.csv | xargs -I {} mv {} ${TARGET_DIR}/
ls results/*.json | xargs -I {} mv {} ${TARGET_DIR}/
