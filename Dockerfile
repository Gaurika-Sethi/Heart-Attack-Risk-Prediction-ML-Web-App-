# -------------------------------
# 1. Base image (Python 3.10)
# -------------------------------
    FROM python:3.10-slim

    # -------------------------------
    # 2. Set working directory
    # -------------------------------
    WORKDIR /app
    
    # -------------------------------
    # 3. Install system dependencies
    # -------------------------------
    RUN apt-get update && apt-get install -y \
        build-essential \
        libopenblas-dev \
        liblapack-dev \
        && rm -rf /var/lib/apt/lists/*
    
    # -------------------------------
    # 4. Copy requirement file
    # -------------------------------
    COPY requirements.txt .
    
    # -------------------------------
    # 5. Upgrade pip + install packages
    # -------------------------------
    RUN pip install --upgrade pip
    RUN pip install -r requirements.txt
    
    # -------------------------------
    # 6. Copy all project files
    # -------------------------------
    COPY . .
    
    # -------------------------------
    # 7. Expose Streamlit port
    # -------------------------------
    EXPOSE 8501
    
    # -------------------------------
    # 8. Streamlit startup command
    # -------------------------------
    ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    
    