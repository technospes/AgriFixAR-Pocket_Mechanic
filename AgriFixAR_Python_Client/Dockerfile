# Use Python 3.11
FROM python:3.11

# Set working directory
WORKDIR /code

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Create cache directory
RUN mkdir -p /code/cache
RUN chmod -R 777 /code

# Install Python libraries (No Cache)
COPY ./requirements.txt /code/requirements.txt
# Added --no-cache-dir to ensure fresh download
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy code
COPY . .

# Expose port
EXPOSE 7860

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]