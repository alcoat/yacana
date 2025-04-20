import os
import mimetypes
from urllib.parse import urlparse
import urllib.request
import base64


class Media:

    @staticmethod
    def get_as_openai_dict(path: str):

        main_type, file_mime = Media._get_file_type(path)

        if main_type == "image":
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{file_mime};base64,{Media._path_to_base64(path)}",
                }}
        elif main_type == "audio":
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": Media._path_to_base64(path),
                    "format": file_mime
            }}
        else:
            raise ValueError(f"Unsupported media type: {file_type}")

    @staticmethod
    def _get_file_type(file_path: str) -> str:
        """
        Determines the file type based on the file extension.
        @return: str : The file type (e.g., 'audio', 'image', etc.) or 'unknown' if not recognized
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            main_type, subtype = mime_type.split('/')
            return main_type, subtype  # Returns both the main type and subtype
        raise ValueError(f"Unsupported file type: {file_path}")


    @staticmethod
    def _path_to_base64(path: str) -> str:
        """
        Converts the content of a file or URL to a base64-encoded string.
        @param path: str : The path to the file or URL
        @return: str : The base64-encoded content
        """
        if Media.is_url(path):
            try:
                with urllib.request.urlopen(path) as response:
                    data = response.read()
            except Exception as e:
                raise ValueError(f"Failed to fetch URL: {e}")
        elif os.path.isfile(path):
            with open(path, "rb") as file:
                data = file.read()
        else:
            raise ValueError(f"Invalid path: {path} is neither a valid URL nor a file on disk.")

        # Convert to base64
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def is_url(path: str) -> bool:
        """
        Checks if the given path is a URL.
        @param path: str : The path to check
        @return: bool : True if the path is a URL, False otherwise
        """
        parsed = urlparse(path)
        return bool(parsed.scheme and parsed.netloc)