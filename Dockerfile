# 1. Start with a lightweight Python 3.10 image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install the packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your files (main.py, the model, etc.) into the container
COPY . .

# 5. Tell Docker what port to expose
EXPOSE 7860

# 6. The command to run when the container starts
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]