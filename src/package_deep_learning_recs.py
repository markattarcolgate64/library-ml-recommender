import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import torch  # Deep learning framework
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
from torch.utils.data import Dataset, DataLoader  # For data handling
from sklearn.model_selection import train_test_split  # For splitting data
import os  # For file operations
import pickle  # For saving/loading Python objects
import json  # For JSON processing
from tqdm import tqdm  # For progress bars
import matplotlib.pyplot as plt  # For visualization


class PackageEmbeddingDataset(Dataset):
    """
    Dataset for training the package embedding model
    """
    def __init__(self, feature_matrix, package_names):
        self.feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32)
        self.package_names = package_names
        self.n_packages = len(package_names)
        
        # Create a mapping from package name to index
        self.package_to_idx = {name: i for i, name in enumerate(package_names)}
    
    def __len__(self):
        return self.n_packages * 5  # Multiple samples per package for better training
    
    def __getitem__(self, idx):
        # Get the anchor package
        anchor_idx = idx % self.n_packages
        anchor_features = self.feature_matrix[anchor_idx]
        
        # Get a positive example (we'll use the same package as positive)
        pos_features = anchor_features
        
        # Get a negative example (a random different package)
        neg_idx = np.random.choice(
            [i for i in range(self.n_packages) if i != anchor_idx]
        )
        neg_features = self.feature_matrix[neg_idx]
        
        return anchor_features, pos_features, neg_features


class PackageEmbeddingModel(nn.Module):
    """
    Neural network model for learning package embeddings
    """
    def __init__(self, input_dim, embedding_dim=64, hidden_dim=128):
        super(PackageEmbeddingModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x):
        return self.encoder(x)
    
    def get_embedding(self, x):
        with torch.no_grad():
            return self.encoder(x).detach().cpu().numpy()
    


class TripletLoss(nn.Module):
    """
    Triplet loss function for training the package embedding model 
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_dist = torch.norm(positive- anchor, dim =1)
        negative_dist = torch.norm(negative - anchor, dim= 1)

        loss = torch.clamp(positive_dist - negative_dist + self.margin, min=0)

        return loss.mean()
    

class DeepLibraryRecommender:
    """
    Deep learning based package reccomender
    """

    def __init__(self, embedding_dim=64, hidden_dim=128, model_dir='/Users/markattar/Desktop/GitHub/library-scraper/lib_recommender/models'):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.model_dir = model_dir
        self.model = None
        self.packages_df = None
        self.feature_matrix = None
        self.package_embeddings = None
    
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def get_feature_cols(self, feature_df):
        # Define feature prefixes in an array for maintainability
        feature_prefixes = ['tfidf_', 'is_', 'has_dep_']
        # Define exact column names to include (optional)
        exact_columns = ['dependency_count']

        # Get feature columns by checking against prefixes array
        feature_cols = [col for col in feature_df.columns 
                    if any(col.startswith(prefix) for prefix in feature_prefixes)
                    or col in exact_columns]
    
        return feature_cols

    def preprocess_data(self, feature_df):
        package_names = feature_df['name'].values
        feature_cols = self.get_feature_cols(feature_df)

        feature_matrix = feature_df[feature_cols].values
        return feature_matrix, package_names, feature_cols 
    
    def _generate_embeddings(self, device):
        self.model.eval()

        features = torch.tensor(self.feature_matrix, dtype=torch.float32, device=device).to(device)

        batch_size = 64
        embeddings = []

        for i in range(0, len(features), batch_size):
            batch = features[i:i+batch_size]
            with torch.no_grad():
                batch_emb = self.model(batch).detach().cpu().numpy()
            embeddings.append(batch_emb)

        self.package_embeddings = np.vstack(embeddings)
        

    def save_model(self, filename='deep_package_recommender'):
        if self.model is None:
            print("Error: no model to save")
            return
        
        model_path = os.path.join(self.model_dir, f"{filename}.pt")
        data_path = os.path.join(self.model_dir, f"{filename}_data.pkl")

        torch.save(self.model.state_dict(), model_path)
        
        #Save package embeddings for query/inference, save packages df for metadata, save feature matrix, save embedding/hidden dim for query
        data = {
        'package_embeddings': self.package_embeddings,
        'packages_df': self.packages_df,
        'feature_matrix': self.feature_matrix,
        'embedding_dim': self.embedding_dim,
        'hidden_dim': self.hidden_dim
        }
    
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model saved to {model_path} and {data_path}")

    def load_model(self, filename='deep_recommender', device=None):
        """
        Load the model and embeddings from disk
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model_path = os.path.join(self.model_dir, f"{filename}_model.pt")
        data_path = os.path.join(self.model_dir, f"{filename}_data.pkl")
        
        try:
            # Load data
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.package_embeddings = data['package_embeddings']
            self.packages_df = data['packages_df']
            self.feature_matrix = data['feature_matrix']
            self.embedding_dim = data['embedding_dim']
            self.hidden_dim = data['hidden_dim']
            
            # Create and load model
            input_dim = self.feature_matrix.shape[1]
            self.model = PackageEmbeddingModel(
                input_dim=input_dim,
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim
            ).to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            
            print(f"Model loaded from {model_path} and {data_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
    
    def query_embedding(self, query_text, feature_extractor, device=None):
        """
        Generate embedding from text query 
        Take in text query, run through feature extractor, generate embedding, return embedding
        """

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.model is None:
            print("Model not loaded. Please load or train a model first")
            return None
        
        dummy_package = pd.DataFrame({
            'name': ['query'],
            'summary': [query_text],
            'description': [query_text],
            'dependencies':[[]],
            'classifiers': [[]],
        })


        processed_package = feature_extractor.process_data(dummy_package)

        feature_cols = self.get_feature_cols(processed_package)

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

    

        #Generate embedding from text query


    def train(self, feature_df, epochs=10, batch_size=64, lr=0.001, test_perc=0.15, val_perc=0.2, device=None):

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'


        #Separate package metadata from feature matrix 
        self.packages_df = feature_df[['name', 'summary', 'description']].copy()

        self.feature_matrix, package_names, feature_cols = self.preprocess_data(feature_df)

        input_dim = len(feature_cols)

        print(input_dim)

        self.model = PackageEmbeddingModel(
            input_dim=input_dim,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        ).to(device)

        dataset = PackageEmbeddingDataset(self.feature_matrix, package_names)
        

        #Splitting the dataset
        val_size = int(val_perc * len(dataset))
        test_size = int(test_perc * len(dataset))
        train_size = int(len(dataset) - val_size - test_size)

        print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}, Total size: {len(dataset)}")

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
         
        #Creating the data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        #Setup loss function & optimizer 
        criterion = TripletLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        train_losses = []
        val_losses = []

        print(f"Training model on {device}")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_batches = 0 

            for anchor, positive, negative in tqdm(train_loader, desc=f"Epoch {epoch +1}/{epochs} (Train)"):
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                print(anchor.shape, positive.shape, negative.shape)
                

                optimizer.zero_grad()

                anchor_embedding = self.model(anchor)
                positive_embedding = self.model(positive)
                negative_embedding = self.model(negative)

                loss = criterion(anchor_embedding, positive_embedding, negative_embedding)

                loss.backward()
                optimizer.step()


                train_loss += loss.item()
                train_batches += 1

            train_loss /= train_batches 
            train_losses.append(train_loss)

            self.model.eval()
            val_loss = 0.0
            val_batches = 0 

            with torch.no_grad():
                for anchor, positive, negative in tqdm(train_loader, desc=f"Epoch {epoch +1}/{epochs} (Train)"):
                    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                    

                    optimizer.zero_grad()

                    anchor_embedding = self.model(anchor)
                    positive_embedding = self.model(positive)
                    negative_embedding = self.model(negative)

                    loss = criterion(anchor_embedding, positive_embedding, negative_embedding)

                    val_loss += loss.item()
                    val_batches += 1

                val_loss /= train_batches 
                val_losses.append(train_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")


        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
        plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(self.model_dir, 'training_loss.png'))

        # Generate embeddings for all packages
        self._generate_embeddings(device)

        return train_losses, val_losses




        

if __name__ == "__main__":
    processed_data_dir = '/Users/markattar/Desktop/GitHub/library-scraper/lib_recommender/data/processed'
    feature_df = pd.read_csv(os.path.join(processed_data_dir, 'processed_packages.csv'))
    recommender = DeepLibraryRecommender()
    recommender.train(feature_df)
        

        