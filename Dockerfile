# Use an official Python runtime as a parent image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Set the working directory
WORKDIR /app

# Install the pinned serving dependencies first so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the service needs
COPY src ./src
COPY app.py .
COPY data/archive.zip ./data/archive.zip
# The picture and sentence vectors (half-size numbers, two files under 50 MB)
COPY artifacts/vectors ./artifacts/vectors

# Run as a non-root user
RUN useradd --create-home --uid 10001 appuser
USER appuser

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME ProductSimilarityApp
# Stop with an error when the vectors are missing, instead of using the baseline search
ENV PS_REQUIRE_VECTORS=1

# The engine is built once at startup (about 3 seconds), then requests are fast
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
