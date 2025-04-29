# Use Mambaforge (fast conda with conda-forge defaults)
FROM condaforge/mambaforge:latest

# Set environment name
ENV ENV_NAME=carbon-env

# Create the environment and install packages
RUN mamba create -n $ENV_NAME \
    python=3.10 \
    streamlit \
    rasterio \
    matplotlib \
    numpy \
    pandas \
    plotly \
    scikit-learn \
    -y

# Use the conda environment
SHELL ["conda", "run", "-n", "carbon-env", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Copy app code into container
COPY . /app

# Expose the Streamlit default port
EXPOSE 8501

# Start the app
CMD ["conda", "run", "--no-capture-output", "-n", "carbon-env", "streamlit", "run", "planet_carbon_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
