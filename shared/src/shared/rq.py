import requests
import time
import logging

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Any, Dict, Union


class Requests:
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
        status_forcelist: tuple = (500, 502, 503, 504, 403, 401, 400),
        timeout: Union[float, tuple] = (5, 15),
    ):
        """
        Initialize CustomRequests with retry configuration.
        
        Args:
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor between retries
            status_forcelist: HTTP status codes to retry on
            timeout: Request timeout (connect timeout, read timeout)
        """
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        
        # Mount the adapter to both HTTP and HTTPS
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.timeout = timeout
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def request(
        self,
        method: str,
        url: str,
        **kwargs: Any
    ) -> requests.Response:
        """
        Make an HTTP request with automatic retry functionality.
        
        Args:
            method: HTTP method
            url: Target URL
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            requests.Response object
        """
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
            
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {str(e)}")
            raise

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """Perform GET request"""
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        """Perform POST request"""
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> requests.Response:
        """Perform PUT request"""
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> requests.Response:
        """Perform DELETE request"""
        return self.request("DELETE", url, **kwargs)

    def close(self):
        """Close the session"""
        self.session.close()
