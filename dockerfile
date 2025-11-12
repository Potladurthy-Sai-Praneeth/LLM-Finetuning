# Stage 1: Builder stage
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310 AS builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy the entire project (including setup.py from root)
COPY . /build/
RUN if [ -f /build/setup.py ]; then \
    cd /build && pip install --no-cache-dir --target=/install .; \
    fi

# Stage 2: Runtime stage
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310 AS runtime

# Set working directory
WORKDIR /app

# Copy only necessary runtime libraries and remove unnecessary files
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /install /usr/local/lib/python3.10/site-packages/

# Copy your custom package source (if needed for imports)
COPY . .

# Clean up unnecessary files to reduce size
RUN find /usr/local/lib/python3.10/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.10/site-packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.10/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.10/site-packages -name "*.pyo" -delete && \
    find /usr/local/lib/python3.10/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Set environment variable for script selection (default: deepspeed_finetune)
ENV TRAIN_SCRIPT=finetune.deepspeed_finetune

# Create entrypoint script
RUN echo '#!/bin/bash\n\
torchrun \\\n\
    --nnodes=1 \\\n\
    --nproc_per_node=auto \\\n\
    --rdzv_backend=c10d \\\n\
    --rdzv_endpoint=localhost:0 \\\n\
    -m ${TRAIN_SCRIPT} "$@"' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Entry point for torchrun with flexible script selection
ENTRYPOINT ["/app/entrypoint.sh"]
