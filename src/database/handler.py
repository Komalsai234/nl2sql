import os
import json
import logging
from typing import Dict, Any
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text
import threading

class DataHandler:
    def __init__(self, db_uri: str, cache_dir: str = "db_cache", cache_file: str = "schema_full.json"):
        self.db_uri = db_uri
        self.engine = create_engine(self.db_uri)
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(self.cache_dir, cache_file)
        self._last_error = threading.local()

    def _load_schema_from_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error reading schema cache: {e}")
        return {}

    def _save_schema_to_cache(self, schema: Dict[str, Any]):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(schema, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving schema cache: {e}")

    def get_schema(self) -> Dict[str, Any]:
        schema = self._load_schema_from_cache()
        if schema:
            return schema

        try:
            inspector = inspect(self.engine)
            schema = {}
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                if not columns:
                    continue
                schema[table_name] = {
                    "columns": {col["name"]: str(col["type"]) for col in columns},
                    "primary_keys": inspector.get_pk_constraint(table_name).get("constrained_columns", []),
                    "foreign_keys": [
                        {
                            "column": fk["constrained_columns"][0],
                            "references": {"table": fk["referred_table"], "column": fk["referred_columns"][0]}
                        }
                        for fk in inspector.get_foreign_keys(table_name)
                        if fk["constrained_columns"] and fk["referred_columns"]
                    ],
                }
            self._save_schema_to_cache(schema)
            return schema
        except Exception as e:
            logging.error(f"Error loading database schema: {e}")
            return {}

    def execute_query(self, query):
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                data = result.fetchall()
                return [dict(row._mapping) for row in data] if data else []
        except SQLAlchemyError as e:
            self._last_error.value = str(e)
            logging.error(f"Database query execution error: {e}")
            return None
        
    def get_last_error(self):
        return getattr(self._last_error, "value", "No error recorded.")
        
    def get_unique_column_values(self, table: str, column: str):

        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table)
            column_info = next((col for col in columns if col["name"] == column), None)
            if not column_info:
                return []

            column_type = str(column_info["type"]).lower()
            if "char" not in column_type:
                return []
            query = text(f"SELECT DISTINCT {column} FROM {table};")
            with self.engine.connect() as connection:
                result = connection.execute(query)
                unique_values = [row[0] for row in result.fetchall() if row[0] is not None]

                return unique_values if len(unique_values) <= 400 else [] 
        except SQLAlchemyError as e:
            logging.error(f"Error fetching unique values from {table}.{column}: {e}")
            return []