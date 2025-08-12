"""
HDFS utilities for file operations
"""
import os
import tempfile
import shutil
from urllib.parse import urlparse
import pyarrow.fs


def upload_local_directory_to_hdfs(local_path, hdfs_path):
    """Upload local directory to HDFS"""
    try:
        print(f"[HDFS UPLOAD] From '{local_path}' to '{hdfs_path}'")
        parsed_uri = urlparse(hdfs_path)
        hdfs = pyarrow.fs.HadoopFileSystem(
            host=parsed_uri.hostname, 
            port=parsed_uri.port
        )
        hdfs.create_dir(parsed_uri.path, recursive=True)
        
        for filename in os.listdir(local_path):
            local_file = os.path.join(local_path, filename)
            hdfs_file = os.path.join(parsed_uri.path, filename)
            
            if os.path.isfile(local_file):
                with open(local_file, "rb") as f_local, \
                     hdfs.open_output_stream(hdfs_file) as f_hdfs:
                    f_hdfs.write(f_local.read())
                    
        print(f"[HDFS UPLOAD] Successfully uploaded to {hdfs_path}")
        
    except Exception as e:
        print(f"[HDFS UPLOAD ERROR] {e}")


def upload_file_to_hdfs(local_file_path, hdfs_file_path):
    """Upload a single file to HDFS"""
    try:
        parsed_uri = urlparse(hdfs_file_path)
        hdfs = pyarrow.fs.HadoopFileSystem(
            host=parsed_uri.hostname, 
            port=parsed_uri.port
        )
        
        # Create parent directory if needed
        parent_path = os.path.dirname(parsed_uri.path)
        hdfs.create_dir(parent_path, recursive=True)
        
        with open(local_file_path, "rb") as f_local, \
             hdfs.open_output_stream(parsed_uri.path) as f_hdfs:
            f_hdfs.write(f_local.read())
            
        print(f"[HDFS UPLOAD] File uploaded: {hdfs_file_path}")
        
    except Exception as e:
        print(f"[HDFS UPLOAD ERROR] {e}")


def download_file_from_hdfs(hdfs_file_path, local_file_path):
    """Download a file from HDFS to local filesystem"""
    try:
        parsed_uri = urlparse(hdfs_file_path)
        hdfs = pyarrow.fs.HadoopFileSystem(
            host=parsed_uri.hostname, 
            port=parsed_uri.port
        )
        
        with hdfs.open_input_stream(parsed_uri.path) as f_hdfs, \
             open(local_file_path, "wb") as f_local:
            f_local.write(f_hdfs.read())
            
        print(f"[HDFS DOWNLOAD] File downloaded: {local_file_path}")
        
    except Exception as e:
        print(f"[HDFS DOWNLOAD ERROR] {e}")
        raise


def cleanup_hdfs_directory(hdfs_path):
    """Clean up HDFS directory"""
    try:
        parsed_uri = urlparse(hdfs_path)
        hdfs = pyarrow.fs.HadoopFileSystem(
            host=parsed_uri.hostname, 
            port=parsed_uri.port
        )
        
        if hdfs.get_file_info(parsed_uri.path).type != pyarrow.fs.FileType.NotFound:
            hdfs.delete_dir(parsed_uri.path)
            print(f"[CLEANUP] Removed HDFS directory: {hdfs_path}")
            
    except Exception as e:
        print(f"[CLEANUP ERROR] Failed to remove {hdfs_path}: {e}")


def create_temp_local_dir():
    """Create a temporary local directory"""
    return tempfile.mkdtemp()


def cleanup_local_dir(local_path):
    """Clean up local directory"""
    try:
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
            print(f"[CLEANUP] Removed local directory: {local_path}")
    except Exception as e:
        print(f"[CLEANUP ERROR] Failed to remove {local_path}: {e}")
