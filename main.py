from src.agents.selector_agent import Selector
from src.database.handler import DataHandler
from src.agents.decomposer_agent import Decomposer
from src.vector_db.create_vectordb import VectorDBHandler
import logging
from src.constant import *
import os
from src.agents.refiner_agent import Refiner


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

data_handler = DataHandler(db_uri=DATABASE_URI)

vector_db_handler = VectorDBHandler(data_handler=data_handler)


selector = Selector(data_handler=data_handler,
                    vector_db_handler=vector_db_handler)

user_query = "How many offers are rolled out in last 5 years in different business units?"

search_results = selector.retrieve_relevant_tables(user_query=user_query)

schema_info = {table: details["columns"] for table, details in search_results.items()}
fk_info = {table: details["foreign_keys"] for table, details in search_results.items() }

print(search_results)

decomposer = Decomposer()

sql_query = decomposer.get_final_sql_from_user_query(user_query=user_query,
                                                     schema_info=schema_info,
                                                     fk_info=fk_info)
print(sql_query)
