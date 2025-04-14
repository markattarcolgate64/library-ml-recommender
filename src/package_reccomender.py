# src/models/recommender.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

class LibraryRecommender:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.packages_df = None
        self.feature_matrix = None
        self.tfidf_vectorizer = None
        self.feature_names = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Download NLTK data if not already downloaded
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except:
            print("Warning: Could not download NLTK data")
        
        self.stopwords = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stopwords and len(token) >= 3
        ]
        
        return ' '.join(cleaned_tokens)
    
    def fit(self, packages_df, text_column='combined_text', max_features=1000):
        """
        Train the recommendation model
        
        Parameters:
        -----------
        packages_df : pandas.DataFrame
            DataFrame containing package information
        text_column : str
            Column name containing the text to vectorize
        max_features : int
            Maximum number of features for TF-IDF vectorizer
        """
        self.packages_df = packages_df.copy()
        
        # If the text column doesn't exist, create it from summary and description
        if text_column not in self.packages_df.columns:
            print(f"Column {text_column} not found, creating from summary and description...")
            self.packages_df['combined_text'] = ''
            
            if 'summary' in self.packages_df.columns:
                self.packages_df['combined_text'] += ' ' + self.packages_df['summary'].fillna('')
            
            if 'description' in self.packages_df.columns:
                self.packages_df['combined_text'] += ' ' + self.packages_df['description'].fillna('')
            
            text_column = 'combined_text'
        
        # Preprocess text if it hasn't been done yet
        if not f'processed_{text_column}' in self.packages_df.columns:
            print(f"Preprocessing {text_column}...")
            self.packages_df[f'processed_{text_column}'] = self.packages_df[text_column].apply(self.preprocess_text)
            text_column = f'processed_{text_column}'
        
        # Create TF-IDF vectorizer
        print("Creating TF-IDF vectors...")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        text_features = self.tfidf_vectorizer.fit_transform(self.packages_df[text_column].fillna(''))
        
        # Store feature names
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Convert sparse matrix to dense for easier manipulation
        self.feature_matrix = text_features.toarray()
        
        print(f"Model fitted with {self.feature_matrix.shape[1]} features")
        return self
    
    def save_model(self, filename='library_recommender'):
        """Save the model to disk"""
        model_path = os.path.join(self.model_dir, f"{filename}.pkl")
        
        # Create a dictionary with all necessary components
        model_data = {
            'packages_df': self.packages_df,
            'feature_matrix': self.feature_matrix,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'feature_names': self.feature_names
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, filename='library_recommender'):
        """Load the model from disk"""
        model_path = os.path.join(self.model_dir, f"{filename}.pkl")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.packages_df = model_data['packages_df']
            self.feature_matrix = model_data['feature_matrix']
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.feature_names = model_data['feature_names']
            
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_recommendations_from_text(self, query_text, n=5):
        """
        Get recommendations based on a text query
        
        Parameters:
        -----------
        query_text : str
            Text description of what the user is looking for
        n : int
            Number of recommendations to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the top n recommended packages
        """
        if self.tfidf_vectorizer is None or self.feature_matrix is None:
            print("Model not fitted yet. Please fit the model first.")
            return None
        
        # Preprocess the query
        processed_query = self.preprocess_text(query_text)
        
        # Transform query to vector
        query_vector = self.tfidf_vectorizer.transform([processed_query]).toarray()
        
        # Calculate similarity with all packages
        similarities = cosine_similarity(query_vector, self.feature_matrix)[0]
        
        # Get indices of top n matches
        top_indices = np.argsort(similarities)[::-1][:n]
        
        # Create results dataframe
        recommendations = self.packages_df.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarities[top_indices]
        
        # Extract key terms that matched
        recommendations['matched_terms'] = recommendations.apply(
            lambda row: self._get_matching_terms(query_vector, row.name), axis=1
        )
        
        return recommendations[['name', 'summary', 'similarity_score', 'matched_terms']]
    
    def get_recommendations_from_package(self, package_name, n=5):
        """
        Get recommendations based on a known package
        
        Parameters:
        -----------
        package_name : str
            Name of the package to find similar packages to
        n : int
            Number of recommendations to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the top n recommended packages
        """
        if self.packages_df is None or self.feature_matrix is None:
            print("Model not fitted yet. Please fit the model first.")
            return None
        
        # Find the package in the dataframe
        package_idx = self.packages_df[self.packages_df['name'].str.lower() == package_name.lower()].index
        
        if len(package_idx) == 0:
            print(f"Package '{package_name}' not found in the dataset.")
            return None
        
        package_idx = package_idx[0]
        
        # Get the package vector
        package_vector = self.feature_matrix[package_idx].reshape(1, -1)
        
        # Calculate similarity with all packages
        similarities = cosine_similarity(package_vector, self.feature_matrix)[0]
        
        # Get indices of top n+1 matches (the package itself will be included)
        top_indices = np.argsort(similarities)[::-1][:n+1]
        
        # Remove the package itself if it's in the results
        top_indices = [idx for idx in top_indices if idx != package_idx][:n]
        
        # Create results dataframe
        recommendations = self.packages_df.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarities[top_indices]
        
        # Extract key terms that matched
        recommendations['matched_terms'] = recommendations.apply(
            lambda row: self._get_matching_terms(package_vector, row.name), axis=1
        )
        
        return recommendations[['name', 'summary', 'similarity_score', 'matched_terms']]
    
    def _get_matching_terms(self, query_vector, row_idx, top_n=5):
        """Get the top matching terms between query and package"""
        # Get the vector for the package
        package_vector = self.feature_matrix[row_idx]
        
        # Calculate term importance (element-wise product)
        term_importance = query_vector[0] * package_vector
        
        # Get indices of top terms
        top_term_indices = np.argsort(term_importance)[::-1][:top_n]
        
        # Get the term names
        top_terms = [self.feature_names[idx] for idx in top_term_indices if term_importance[idx] > 0]
        
        return ', '.join(top_terms)

# Example usage
if __name__ == "__main__":
    # Load processed data
    try:
        processed_file = "data/processed/processed_packages.csv"
        processed_df = pd.read_csv(processed_file)
        
        # Train recommender
        recommender = LibraryRecommender()
        recommender.fit(processed_df)
        
        # Save model
        recommender.save_model()
        
        # Test recommendations
        query = "I need a library for data visualization that works well with pandas"
        recommendations = recommender.get_recommendations_from_text(query)
        print("\nRecommendations for query:")
        print(query)
        print(recommendations)
        
        # Test package-based recommendations
        package_recs = recommender.get_recommendations_from_package("pandas")
        print("\nSimilar packages to pandas:")
        print(package_recs)
        
    except Exception as e:
        print(f"Error in recommendation model: {e}")