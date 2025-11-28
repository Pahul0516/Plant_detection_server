# Use an official PyTorch image with CPU support (matches Python 3.11)
FROM pytorch/pytorch:latest

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy the project source
COPY . /app

# Expose the port used by Uvicorn
EXPOSE 8000

# Ensure logs are printed immediately
ENV PYTHONUNBUFFERED=1

# Default command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]