FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

LABEL description="3D-PMGNet LiTS liver tumor segmentation"

RUN pip install --no-cache-dir \
    monai>=1.3 \
    nibabel>=5.0 \
    SimpleITK>=2.3 \
    tqdm>=4.65 \
    matplotlib>=3.7 \
    timm>=0.9 \
    tensorboard>=2.13

WORKDIR /workspace
COPY . .

CMD ["bash"]
