import os
from pathlib import Path
import requests
from tqdm.auto import tqdm

def download_file(url: str, output_dir: str = '.') -> str:
    """
    Download a file from URL with progress bar showing proper units.
    
    Args:
        url: URL to download from
        output_dir: Directory to save the file (defaults to current directory)
    
    Returns:
        str: Absolute path to the downloaded file
    """
    # Extract filename from URL or use default
    filename = url.split('/')[-1] or 'model.pt'
    if not filename.endswith('.pt'):
        filename += '.pt'
    
    # Convert to absolute path
    output_path = Path(output_dir).resolve() / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize download
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get file size if available
    total_size = int(response.headers.get('content-length', 0))
    
    # Configure progress bar
    progress_kwargs = {
        'desc': f'Downloading {filename}',
        'unit': 'iB',
        'unit_scale': True,
        'unit_divisor': 1024,
        'miniters': 1,
        'total': total_size if total_size else None
    }
    
    # Download with progress bar
    with open(output_path, 'wb') as file, tqdm(**progress_kwargs) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                size = file.write(chunk)
                pbar.update(size)
    
    # Return absolute path as string
    return str(output_path.absolute())

if __name__ == '__main__':
    # Example usage
    url = 'https://huggingface.co/TerminatorPower/EzeLLM-base-text-fp32/resolve/main/model.pt'
    try:
        file_path = download_file(url)
        print(f"\nDownload completed. Absolute path: {file_path}")
        
        # Example of using the absolute path
        print(f"File exists: {os.path.exists(file_path)}")
        print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Error downloading file: {str(e)}")