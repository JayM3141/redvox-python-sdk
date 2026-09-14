import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class CloudUploader:
    """Handles uploading exported files (HDF5/NetCDF) to cloud providers."""

    @staticmethod
    def upload_to_s3(file_path: str, bucket_name: str, object_name: Optional[str] = None) -> bool:
        """
        Uploads a file to an AWS S3 bucket.
        Requires AWS credentials in environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        or IAM roles.
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            logger.error("boto3 is not installed.")
            return False

        if object_name is None:
            object_name = os.path.basename(file_path)

        s3_client = boto3.client('s3')
        try:
            s3_client.upload_file(file_path, bucket_name, object_name)
            logger.info(f"Successfully uploaded {file_path} to s3://{bucket_name}/{object_name}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading to S3: {e}")
            return False

    @staticmethod
    def upload_to_gcp(file_path: str, bucket_name: str, object_name: Optional[str] = None) -> bool:
        """
        Uploads a file to a Google Cloud Storage bucket.
        Requires GCP credentials in environment (GOOGLE_APPLICATION_CREDENTIALS).
        """
        try:
            from google.cloud import storage
        except ImportError:
            logger.error("google-cloud-storage is not installed.")
            return False

        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(object_name)

            blob.upload_from_filename(file_path)
            logger.info(f"Successfully uploaded {file_path} to gs://{bucket_name}/{object_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to GCP: {e}")
            return False
