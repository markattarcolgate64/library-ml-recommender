# src/preprocessing/feature_extraction.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os

class PackageFeatureExtractor:
    def __init__(self, download_nltk_data=True):
        if download_nltk_data:
            # Download required NLTK data
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
            except:
                print("Warning: Could not download NLTK data. Preprocessing may be affected.")
        
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.vectorizers = {}
    
    def preprocess_text(self, text, min_length=3):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and short words, and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stopwords and len(token) >= min_length
        ]
        
        return ' '.join(cleaned_tokens)
    
    def extract_features(self, df):
        """Extract features from package data"""
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Preprocess text columns
        text_cols = ['summary', 'description', 'keywords']
        for col in text_cols:
            if col in processed_df.columns:
                print(f"Preprocessing {col}...")
                processed_df[f'processed_{col}'] = processed_df[col].apply(self.preprocess_text)
        
        # Create a combined text field for vectorization
        processed_df['combined_text'] = ''
        for col in text_cols:
            if f'processed_{col}' in processed_df.columns:
                processed_df['combined_text'] += ' ' + processed_df[f'processed_{col}'].fillna('')
        
        # Extract features from dependencies
        if 'dependencies' in processed_df.columns:
            # Count dependencies
            processed_df['dependency_count'] = processed_df['dependencies'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            
            # Top dependencies as features
            all_deps = []
            for deps in processed_df['dependencies']:
                if isinstance(deps, list):
                    all_deps.extend(deps)
            
            # Get top dependencies
            top_deps = pd.Series(all_deps).value_counts().head(50).index.tolist()
            
            # Create binary features for top dependencies
            for dep in top_deps:
                processed_df[f'has_dep_{dep}'] = processed_df['dependencies'].apply(
                    lambda x: 1 if isinstance(x, list) and dep in x else 0
                )
        
        # Extract features from classifiers
        if 'classifiers' in processed_df.columns:
            # Process framework information
            processed_df['is_web_framework'] = processed_df['classifiers'].apply(
                lambda x: 1 if isinstance(x, list) and any('Framework :: Django' in c or 'Framework :: Flask' in c for c in x) else 0
            )
            
            # Process development status
            processed_df['is_stable'] = processed_df['classifiers'].apply(
                lambda x: 1 if isinstance(x, list) and any('Development Status :: 5 - Production/Stable' in c for c in x) else 0
            )
            
            # Topic extraction
            processed_df['is_ml_package'] = processed_df['classifiers'].apply(
                lambda x: 1 if isinstance(x, list) and any('Topic :: Scientific/Engineering :: Artificial Intelligence' in c for c in x) else 0
            )
            
            processed_df['is_data_science'] = processed_df['classifiers'].apply(
                lambda x: 1 if isinstance(x, list) and any('Topic :: Scientific/Engineering :: Information Analysis' in c for c in x) else 0
            )
        
        return processed_df
    
    def vectorize_text(self, df, column='combined_text', max_features=1000):
        """Convert text to TF-IDF vectors"""
        if column not in df.columns:
            print(f"Column {column} not found in dataframe")
            return None
        
        # Create and fit vectorizer
        vectorizer = TfidfVectorizer(max_features=max_features)
        text_vectors = vectorizer.fit_transform(df[column].fillna(''))
        
        # Store vectorizer for later use
        self.vectorizers[column] = vectorizer
        
        # Convert to dataframe
        vector_df = pd.DataFrame(
            text_vectors.toarray(),
            columns=[f'tfidf_{i}' for i in range(text_vectors.shape[1])],
            index=df.index
        )
        
        return vector_df
    
    def process_data(self, df, max_features=500): 
        """Complete preprocessing pipeline"""
        print("Extracting features...")
        processed_df = self.extract_features(df)
        
        print("Vectorizing text...")
        text_vectors = self.vectorize_text(processed_df, max_features=max_features)
        
        if text_vectors is not None:
            # Combine numeric features with text vectors
            numeric_cols = [
                col for col in processed_df.columns 
                if col.startswith('is_') or col.startswith('has_dep_') or col == 'dependency_count'
            ]
            
            final_df = pd.concat([processed_df[['name', 'summary', 'description'] + numeric_cols], text_vectors], axis=1)
            return final_df
        else:
            return processed_df

def get_embedding_for_text(self, query_text, feature_extractor, device=None):
    """
    Generate embedding for a text query
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if self.model is None:
        print("Model not loaded. Please load or train a model first.")
        return None
    
    # Create a dummy package with the query text
    dummy_package = pd.DataFrame({
        'name': ['query'],
        'summary': [query_text],
        'description': [query_text],
        'dependencies': [[]],
        'classifiers': [[]]
    })
    
    # Process the dummy package
    processed_package = feature_extractor.process_data(dummy_package)
    
    # Get feature columns only
    feature_cols = [col for col in processed_package.columns 
                if col.startswith('tfidf_') 
                or col.startswith('is_') 
                or col.startswith('has_dep_')
                or col == 'dependency_count']
    
    # Ensure all feature columns from training are present
    for col in self.feature_matrix.shape[1] - len(feature_cols):
        if f'tfidf_{col}' not in feature_cols:
            processed_package[f'tfidf_{col}'] = 0.0
    
    # Extract features and convert to tensor
    query_features = torch.tensor(processed_package[feature_cols].values, dtype=torch.float32).to(device)
    
    # Generate embedding
    self.model.eval()
    with torch.no_grad():
        query_embedding = self.model(query_features).detach().cpu().numpy()
    
    return query_embedding[0]


# Example usage
if __name__ == "__main__":
    # Load data
    try:
        # Change this path to your actual data file


        input_dir = "/Users/markattar/Desktop/GitHub/library-scraper/lib_recommender/data/raw"
        output_dir = "/Users/markattar/Desktop/GitHub/library-scraper/lib_recommender/data/processed"
        dfs = []

        for f in os.listdir(input_dir):
            if f.endswith(".csv"):
                csv_file = os.path.join(input_dir, f)
                dfs.append(pd.read_csv(csv_file))
            
        packages_df = dfs[0]
        # Process dependencies (convert from string representation back to list)
        if 'dependencies' in packages_df.columns and packages_df['dependencies'].dtype == 'object':
            packages_df['dependencies'] = packages_df['dependencies'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
            )
        

        # Process data
        extractor = PackageFeatureExtractor()
        processed_data = extractor.process_data(packages_df)

        print(processed_data.columns)
        
        # Save processed data
        processed_data.to_csv(os.path.join(output_dir, "processed_packages.csv"), index=False)
        print(f"Processed data shape: {processed_data.shape}")
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")