# Stage 1: Builder stage
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu124.py310 AS builder

# Set working directory
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# This places your 'finetune' module into the system's site-packages
RUN pip install --no-cache-dir .

# Stage 2: Runtime stage
# FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu124.py310 AS runtime

# COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
# COPY --from=builder /usr/local/bin /usr/local/bin

# # Set working directory
# WORKDIR /app

ENTRYPOINT ["python3", "-m", "torch.distributed.run"]

CMD ["--help"]