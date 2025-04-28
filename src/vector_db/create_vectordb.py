import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.database.handler import DataHandler


class VectorDBHandler:
    def __init__(self, data_handler: DataHandler,
                 index_dir: str = "faiss_index",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 index_type: str = "hnsw", 
                 hnsw_m: int = 64): 

        self.data_handler = data_handler
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, f"schema_index_{index_type}.faiss")
        self.table_names_path = os.path.join(index_dir, "table_names.json")
        self.embedding_model_name = embedding_model 
        self.index_type = index_type.lower()
        self.hnsw_m = hnsw_m

        print(f"🤖 Initializing SentenceTransformer model: {self.embedding_model_name}")
        self.sbert_model = SentenceTransformer(self.embedding_model_name)
        self.dimension = self.sbert_model.get_sentence_embedding_dimension()

        self.index = None
        self.table_names = []
        os.makedirs(self.index_dir, exist_ok=True)


        self.load_or_create_vector_db()

    def load_or_create_vector_db(self):
        """Loads an existing FAISS index or creates a new one if not found."""
        if os.path.exists(self.index_path) and os.path.exists(self.table_names_path):
            print(f"🔄 Loading existing FAISS index ({self.index_type}) from {self.index_path}...")
            try:
                self.index = faiss.read_index(self.index_path)
                self.table_names = self._load_table_names()
                print(f"✅ Index loaded successfully with {self.index.ntotal} vectors.")
                if not self.table_names or self.index.ntotal != len(self.table_names):
                     print(f"⚠️ Warning: Index size ({self.index.ntotal}) mismatch with table names ({len(self.table_names)}). Rebuilding index.")
                     self.create_vector_db()

            except Exception as e:
                print(f"❌ Error loading index: {e}. Rebuilding index.")
                self.create_vector_db()
        else:
            print(f"⚡ Index file not found at {self.index_path}. Creating new FAISS index...")
            self.create_vector_db()
        return self.index

    def create_vector_db(self):
        """Creates and saves a new FAISS index based on the database schema."""
        print("📜 Fetching database schema...")
        schema = self.data_handler.get_schema()
        if not schema:
            print("❌ Error: No schema data found. Cannot create vector DB.")

            self.index = None
            self.table_names = []
            return

        print("📝 Formatting schema information...")
        docs, self.table_names = self._format_schema(schema)

        if not docs:
            print("❌ Error: No documents generated from schema. Cannot create vector DB.")
            self.index = None
            self.table_names = []
            return

        print(f"🧠 Encoding {len(docs)} schema documents using '{self.embedding_model_name}'...")
        embeddings = self.sbert_model.encode(docs, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype='float32') # Ensure correct dtype

        print(f"🛠️ Building FAISS index (Type: {self.index_type})...")
        if self.index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == 'flat':
            self.index = faiss.IndexFlatIP(self.dimension) 
        else:
            raise ValueError(f"Unsupported index_type: {self.index_type}. Choose 'flat' or 'hnsw'.")

        self.index.add(embeddings)

        print(f"💾 Saving FAISS index to {self.index_path} ({self.index.ntotal} vectors)...")
        faiss.write_index(self.index, self.index_path)
        self._save_table_names()
        print("✅ New index created and saved successfully.")


    def search(self, user_query: str, top_k: int = 5):
        """
        Searches the FAISS index for tables relevant to the user query.

        Args:
            user_query: The natural language query from the user.
            top_k: The number of top relevant table names to return.

        Returns:
            A list of the top_k most relevant table names.
        """
        if self.index is None or self.index.ntotal == 0:
            print("⚠️ Index is not loaded or is empty. Cannot perform search.")
            if not self.load_or_create_vector_db() or self.index is None or self.index.ntotal == 0:
                 print("❌ Failed to load or create a valid index. Search aborted.")
                 return []

        print(f"🔍 Encoding query and searching for top {top_k} relevant tables...")
        query_embedding = self.sbert_model.encode([user_query], normalize_embeddings=True)
        query_embedding = np.asarray(query_embedding, dtype='float32')

        search_k = top_k 
        distances, indices = self.index.search(query_embedding, search_k)

        results = []
        if indices.size > 0:
            valid_indices = indices[0]
            valid_indices = valid_indices[valid_indices != -1]
            results = [self.table_names[idx] for idx in valid_indices[:top_k]] 

        print(f"📊 Found relevant tables: {results}")
        return results


    def _format_schema(self, schema):
        """Formats the schema into text documents for embedding (Unchanged)."""
        docs = []
        table_names = []
        for table, details in schema.items():
            columns_str = ', '.join(details.get('columns', {}).keys())
            description = details.get('description', 'No description available.')
            text = f"Table: {table}\nDescription: {description}\nColumns: {columns_str}\n"
            docs.append(text)
            table_names.append(table)
        return docs, table_names

    def _save_table_names(self):
        with open(self.table_names_path, "w") as f:
            json.dump(self.table_names, f, indent=4)

    def _load_table_names(self):
        if os.path.exists(self.table_names_path):
            try:
                with open(self.table_names_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"❌ Error reading table names file: {self.table_names_path}. Returning empty list.")
                return []
        return []