# src/scrapers/pypi_scraper.py
import requests
import json
import pandas as pd
import time
from datetime import datetime
import os

class PyPIScraper:
    def __init__(self, output_dir='lib_recommender/data/raw'):
        self.base_url = "https://pypi.org/pypi"
        self.search_url = "https://pypi.org/search/"
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def get_package_info(self, package_name):
        """Get detailed information about a specific package"""
        try:
            url = f"{self.base_url}/{package_name}/json"
            response = requests.get(url)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to fetch data for {package_name}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching package {package_name}: {e}")
            return None
    
    def get_top_packages(self, num_packages=100):
        """
        Get top packages from PyPI based on download counts
        Note: This is a simplified version, as PyPI doesn't provide an official
        API for download rankings. For a full project, consider using 
        https://hugovk.github.io/top-pypi-packages/ or similar data sources.
        """
        # For this demo, we'll use a list of popular packages
        popular_packages = [
            "numpy", "pandas", "matplotlib", "scikit-learn", "tensorflow", 
            "pytorch", "keras", "scipy", "seaborn", "plotly", "requests", 
            "beautifulsoup4", "flask", "django", "fastapi", "transformers", 
            "nltk", "spacy", "gensim", "pillow", "opencv-python", "pytest",
            "tqdm", "streamlit", "dash", "sqlalchemy", "pyspark", "boto3",
            "psycopg2", "pymongo", "redis", "celery", "airflow", "dask",
            "xgboost", "lightgbm", "catboost", "statsmodels", "pyramid",
            "tornado", "scrapy", "selenium", "jupyterlab", "ipython"
        ]
        
        return popular_packages[:min(num_packages, len(popular_packages))]
    
    def collect_package_data(self, num_packages=50, delay=1):
        """Collect data for multiple packages and save to CSV"""
        packages = self.get_top_packages(num_packages)
        all_data = []
        
        for i, package in enumerate(packages):
            print(f"Fetching data for {package} ({i+1}/{len(packages)})")
            package_data = self.get_package_info(package)
          
            if package_data:
                # Extract relevant information
                info = package_data.get('info', {})
                # Get latest release info
                releases = package_data.get('releases', {})
                latest_version = info.get('version', '')
                latest_release = releases.get(latest_version, [{}])[0] if latest_version and latest_version in releases else {}
                
                # Extract dependencies
                requires_dist = info.get('requires_dist', [])
                dependencies = [dep.split(' ')[0] for dep in requires_dist] if requires_dist else []
                
                # Clean data
                description = info.get('description', '')
                if len(description) > 5000:  # Truncate very long descriptions
                    description = description[:5000]
                
                package_info = {
                    'name': info.get('name', ''),
                    'version': latest_version,
                    'summary': info.get('summary', ''),
                    'description': description,
                    'author': info.get('author', ''),
                    'author_email': info.get('author_email', ''),
                    'license': info.get('license', ''),
                    'keywords': info.get('keywords', ''),
                    'homepage': info.get('home_page', ''),
                    'project_url': info.get('project_url', ''),
                    'release_date': latest_release.get('upload_time', ''),
                    'python_version': info.get('requires_python', ''),
                    'dependencies': dependencies,
                    'classifiers': info.get('classifiers', []),
                    'download_url': latest_release.get('url', '')
                }
                
                all_data.append(package_info)
            
            # Be nice to the server
            time.sleep(delay)
        
        # Convert to DataFrame and save
        if all_data:
            df = pd.DataFrame(all_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print()
            filename = os.path.join(self.output_dir, f"pypi_packages_{timestamp}.csv")
            df.to_csv(filename, index=False)
            print(f"Saved data to {filename}")
            
            # Also save raw JSON for further processing
            json_filename = os.path.join(self.output_dir, f"pypi_packages_raw_{timestamp}.json")
            with open(json_filename, 'w') as f:
                json.dump(all_data, f)
            
            return df
        else:
            print("No data collected")
            return None

# Example usage
if __name__ == "__main__":
    scraper = PyPIScraper()
    df = scraper.collect_package_data(num_packages=20)  # Start with a small number for testing
    print(df.head())