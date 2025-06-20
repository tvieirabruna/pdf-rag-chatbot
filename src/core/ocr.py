import os
from mpxpy.mathpix_client import MathpixClient
from mpxpy.errors import MathpixClientError, ConversionIncompleteError

import logging

# Configure mathpix logger to only show errors
logger = logging.getLogger()
logging.getLogger('mathpix').setLevel(logging.ERROR)

class MathpixOCR:
    def __init__(self):
        self.client = MathpixClient(
            app_id=os.getenv("MATHPIX_APP_ID"),
            app_key=os.getenv("MATHPIX_APP_KEY"),
            api_url=os.getenv("MATHPIX_API_URL")
        )

    def pdf_to_markdown(self, file_path: str):
        """
        Converts a PDF file to markdown format using Mathpix API.

        Args:
            file_path (str): Path of the PDF file to convert

        Returns:
            str: The converted markdown content

        Raises:
            Exception: If conversion fails or times out
        """
        try:
            pdf = self.client.pdf_new(
                file_path=file_path,
                convert_to_md=True
            )
            pdf.wait_until_complete(timeout=120)
            markdown = pdf.to_md_text()
            pdf_json = pdf.to_lines_json()
            return markdown, pdf_json
        
        except ConversionIncompleteError:
            raise Exception("Conversion is not yet complete. Try again later.")
        except MathpixClientError as e:
            raise Exception(f"Mathpix error: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")