# Use NVIDIA's NGC PyTorch container with CUDA 12.4 support
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Prevent python to write pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent stderr and stdout output
ENV PYTHONUNBUFFERED=1 

# Set environment variables for UTF-8 support
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Create a directory for the project
RUN mkdir -p /usr/src/

# Set the working directory inside the container
WORKDIR /usr/src/

# Copy the project files
COPY . /usr/src/

# Install the required Python packages with specified versions
RUN pip install -r requirements.txt

# Entrypoint
CMD [" "]

