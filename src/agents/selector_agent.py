import logging

class Selector:
    def __init__(self, data_handler, vector_db_handler):
        self.data_handler = data_handler
        self.vector_db_handler = vector_db_handler

    def retrieve_relevant_tables(self, user_query: str, top_k: int = 15):

        db_schema = self.data_handler.get_schema()

        if not user_query or not isinstance(user_query, str):
            logging.error("Invalid query. Please provide a valid string.")
            return []

        try:
            relevant_tables = self.vector_db_handler.search(user_query, top_k=top_k)

            relevant_schema = {table: db_schema[table] for table in relevant_tables}

            final_schema = {}
            for table, details in relevant_schema.items():
                if table in db_schema:
                    final_columns = {}

                    for column in details.get("columns", []):
                        if column not in db_schema[table]["columns"]:
                            continue  

                        column_type = db_schema[table]["columns"][column]
                        unique_values = self.data_handler.get_unique_column_values(table, column)

        
                        if unique_values and len(unique_values) < 400:
                            final_columns[column] = {
                                "data_type": column_type,
                                "unique_values": unique_values
                            }
                        else:
                            final_columns[column] = column_type  

                    final_schema[table] = {
                        "description": db_schema[table]["description"],
                        "columns": final_columns,
                        "primary_keys": db_schema[table]["primary_keys"],
                        "foreign_keys": db_schema[table]["foreign_keys"],
                    }

            return final_schema

        except Exception as e:
            logging.error(f"Error retrieving relevant tables: {str(e)}")
            return []
