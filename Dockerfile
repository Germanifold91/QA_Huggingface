# Use an official Miniconda3 as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Create a Conda environment using the environment.yml file
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "huggingface_env", "/bin/bash", "-c"]

# Set environment variable
ENV QUESTION="Your question here"

# Make port 80 available to the world outside this container
EXPOSE 80

# Copy the shell script into the Docker container
COPY run_scripts.sh /app/run_scripts.sh

# Make the shell script executable
RUN chmod +x /app/run_scripts.sh

# Run the shell script when the container launches
CMD ["/app/run_scripts.sh"]