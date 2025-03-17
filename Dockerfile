# Stage 1: Export dependencies using Poetry
FROM python:3.9-slim as exporter

# Install Poetry
RUN pip install --no-cache-dir poetry

WORKDIR /app
# Copy only the files needed for dependency resolution
COPY pyproject.toml poetry.lock ./

# Export production dependencies to requirements.txt
RUN poetry export -f requirements.txt --without-hashes --only main -o requirements.txt
# Optionally, to export dev dependencies too, you could run another export:
# RUN poetry export -f requirements.txt --without-hashes --with dev -o requirements-dev.txt

# Stage 2: Build the final image
FROM python:3.9-slim

WORKDIR /app

# Copy the exported requirements file(s) from the exporter stage
COPY --from=exporter /app/requirements.txt .

# Install dependencies using pip and the exported requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy your full application code
COPY . .

# Set default command (adjust as needed for your application)
CMD ["python", "main.py"]