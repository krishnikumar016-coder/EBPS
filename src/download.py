import os
import urllib.request

def download_file(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"File {dest_path} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        # Define headers to prevent simple bot blocking if any
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Successfully downloaded to {dest_path}")
    except Exception as e:
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            print(f"Download failed ({e}), but local file {dest_path} exists. Proceeding with existing file.")
        else:
            print(f"Failed to download {url}: {e}")
            raise e

def main():
    base_url = "https://raw.githubusercontent.com/AmanMalhotra123/HackerEarth_Employees_Burning_Out_Rank_09/master/"
    files = ["train.csv", "test.csv"]
    
    # Target directory is SWE_PROJECT/data
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest_dir = os.path.join(project_dir, "data")
    
    for filename in files:
        url = base_url + filename
        dest_path = os.path.join(dest_dir, filename)
        download_file(url, dest_path)

if __name__ == "__main__":
    main()
