import json
import logging
import re
from sqlalchemy import create_engine, text
from sqlglot import parse_one
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from src.database.handler import DataHandler
from src.constant import *


class Refiner:
    def __init__(self,data_handler:DataHandler):
        self.llm = ChatOpenAI(model_name="o3-2025-04-16",
                              openai_api_key=OPENAI_API_KEY)

        
        self.data_handler = data_handler

    def is_valid_sql(self, sql_query: str):
        try:
            parse_one(sql_query)
            return True
        except Exception as e:
            logging.error(f"SQL Syntax Error: {e}")
            return False

    def refine_sql(self, user_query: str, schema_info: dict, fk_info: dict, sql_query: str, issue: str):
        stepwise_refinement_prompt = """
        You are an expert SQL assistant.

        The user originally asked: "{query}"
        The relevant database schema is:
        {schema_info}

        The foreign key relationships in the schema are:
        {fk_info}

        The initial SQL query is:
        ```sql
        {sql_query}
        ```

        However, the following issue was encountered: "{issue}"

        **Follow this stepwise Chain-of-Thought refinement process:**
        1️⃣ Identify the root cause of the issue (syntax error, missing conditions, incorrect joins, etc.).
        2️⃣ Determine necessary JOINs and conditions using the provided foreign key relationships.
        3️⃣ Adjust table references, filters, and constraints as needed.
        4️⃣ Construct a refined SQL query that resolves the issue while keeping it optimal.

        **Return only the refined SQL query** in JSON format:
        ```json
        {{"refined_sql": "<corrected SQL query>"}}
        ```
        """

        prompt = stepwise_refinement_prompt.format(
            query=user_query,
            schema_info=json.dumps(schema_info, indent=2),
            fk_info=json.dumps(fk_info, indent=2) if fk_info else "No foreign key info available.",
            sql_query=sql_query,
            issue=issue
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()

            clean_response = re.sub(r"```json\s*([\s\S]+?)\s*```", r"\1", response_text).strip()
            refined_sql = json.loads(clean_response).get("refined_sql", "")

            if refined_sql and self.is_valid_sql(refined_sql):
                return refined_sql
            else:
                logging.error("Invalid SQL generated. Falling back to the original SQL.")
                return sql_query

        except json.JSONDecodeError:
            logging.error("Failed to parse JSON from LLM response.")
            return sql_query
        except Exception as e:
            logging.error(f"Error refining SQL: {e}")
            return sql_query

    def refine_and_execute(self, user_query: str, schema_info: dict, fk_info: dict, sql_query: str, max_refinements: int = 3):
        
        schema_info_str = json.dumps(schema_info, indent=2)
        fk_info_str = json.dumps(fk_info, indent=2) if fk_info else "No foreign key info available."
        
        refinement_attempts = 0

        while refinement_attempts < max_refinements:

            if not self.is_valid_sql(sql_query):
                logging.warning("SQL syntax is invalid. Refining query...")
                sql_query = self.refine_sql(user_query, schema_info_str, fk_info_str, sql_query, "Syntax Error")
                refinement_attempts += 1
                continue
            
            test_result = self.data_handler.execute_query(sql_query)

            if test_result is None:
                logging.info("Execution error occurred. Extracting error details...")
                sql_error = self.data_handler.get_last_error()
                sql_query = self.refine_sql(user_query, schema_info_str, fk_info_str, sql_query, f"Execution Error: {sql_error}")
                refinement_attempts += 1
                continue

            if not test_result:
                logging.info("Query executed successfully but returned no data. Refining query...")
                sql_query = self.refine_sql(user_query, schema_info_str, fk_info_str, sql_query, "Query returned no data")
                refinement_attempts += 1
                continue

            return {"sql": sql_query, "data": test_result, "error": None}

        return {"sql": sql_query, "data": None, "error": "Query could not be refined successfully"}
