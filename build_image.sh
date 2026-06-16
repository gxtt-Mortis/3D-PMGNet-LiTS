#!/bin/bash
# 构建 3D-PMGNet Docker 镜像并导出 tar

IMAGE_NAME="3d-pmgnet-lits"
IMAGE_TAG="v1.0"

echo ">>> 构建镜像..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo ">>> 导出镜像..."
docker save -o ${IMAGE_NAME}.tar ${IMAGE_NAME}:${IMAGE_TAG}

echo ">>> 完成: $(ls -lh ${IMAGE_NAME}.tar)"
