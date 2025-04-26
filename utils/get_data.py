import os
import shutil
import tarfile
import zipfile

import requests


def download_file(url, dest_path):
    """Download file from a URL to a destination path"""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {url} to {dest_path}")
    else:
        print(f"Failed to download from {url}, status code: {response.status_code}")

def unzip_file(zip_path, extract_to='.'):
    """Unzip a zip file to the specified directory"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted {zip_path} to {extract_to}")
            return zip_ref.namelist()[0]  # Return the name of the extracted folder
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file.")
        return None

def untar_file(tar_path, extract_to='.'):
    """Extract tar.gz or tar file"""
    try:
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
            print(f"Extracted {tar_path} to {extract_to}")
            return tar_ref.getnames()[0]  # Return the name of the extracted folder
    except tarfile.ReadError:
        print(f"Error: {tar_path} is not a valid tar file.")
        return None

def extract_file(file_path, extract_to='.'):
    """Handle different file types (zip, tar.gz, etc.)"""
    if file_path.endswith('.zip'):
        return unzip_file(file_path, extract_to)
    elif file_path.endswith('.tar.gz') or file_path.endswith('.tar'):
        return untar_file(file_path, extract_to)
    else:
        print(f"Unknown file format for {file_path}")
        return None

def rename_folder(old_name, new_name):
    """Rename a folder"""
    if os.path.isdir(old_name):
        os.rename(old_name, new_name)
        print(f"Renamed folder {old_name} to {new_name}")
    else:
        print(f"Error: {old_name} is not a directory.")

# File URLs and paths
url_2 = "https://figshare.com/ndownloader/files/9746881" # ECG 744 Fragments
url_1 = "https://figshare.com/ndownloader/files/9746845" # ECG 1000 Fragments

file_1 = "dataset_1.zip"
file_2 = "dataset_2.zip"

# Download the datasets
download_file(url_1, file_1)
download_file(url_2, file_2)

# Extract the datasets and rename folders
extracted_folder_1 = extract_file(file_1)
if extracted_folder_1:
    rename_folder(extracted_folder_1, 'Dataset_1') 

extracted_folder_2 = extract_file(file_2)
if extracted_folder_2:
    rename_folder(extracted_folder_2, 'Dataset_2')

# Clean up the zip files after extraction
os.remove(file_1)
os.remove(file_2)
