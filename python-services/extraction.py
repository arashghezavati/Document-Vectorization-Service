from unstructured.partition.auto import partition
import os

def extract_text(file_path):
    """Extract text from a document file.
    
    Supports common document formats:
    1. Documents:
       - PDF (.pdf)
       - Word (.docx)
    
    2. Text and Data:
       - Plain Text (.txt)
       - Markdown (.md)
       - JSON (.json)
       - XML (.xml)
       - HTML (.html, .htm)
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        str: Extracted text from the document
        
    Raises:
        ValueError: If there's an error processing the document or unsupported format
    """
    try:
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # List of supported extensions
        supported_extensions = [
            '.txt', '.pdf', '.docx', 
            '.md', '.json', '.xml',
            '.html', '.htm'
        ]

        # Check if file type is supported
        if ext not in supported_extensions:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats are: {', '.join(supported_extensions)}")

        # Extract text using unstructured
        elements = partition(filename=file_path)
        
        # Join all text elements with newlines
        text = "\n".join([str(element) for element in elements])
        
        # Clean up any double newlines and extra spaces
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        
        if not text.strip():
            raise ValueError("No text was extracted from the document")
            
        return text
        
    except Exception as e:
        raise ValueError(f"Error extracting text from {file_path}: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = extract_text(sys.argv[1])
        print(text)
