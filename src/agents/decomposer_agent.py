import json
import logging
import re
from sqlglot import parse_one
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from src.constant import *
import json5

class Decomposer:

    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4.1-2025-04-14",
            openai_api_key=OPENAI_API_KEY,
            temperature=0
        )

    def validate_sql(self, sql_query):
        try:
            parse_one(sql_query)
            return True
        except Exception as e:
            logging.error(f"SQL Syntax Error: {e}")
            return False

    def standardize_input(self, schema_info, fk_info):
        return json.dumps(schema_info, sort_keys=True, indent=2), json.dumps(fk_info, sort_keys=True, indent=2)

    def decompose_query(self, user_query, schema_info, fk_info):

        stepwise_prompt = """
        You are an expert Postgres SQL assistant.

        The user asked the following question:  
        "{query}"

        The database provides the following information:

        ### 🔹 Top 15 Relevant Tables (Schema Info):
        {schema_info}

        ### 🔹 Foreign Key Relationships:
        {fk_info}

        ---

        ## 🔍 Step 0: Ambiguity & Multiple Interpretation Detection

        Before writing any SQL, determine whether the user query is ambiguous or open to multiple interpretations based on the provided schema.

        Carefully analyze all 15 tables to identify:
        - Overlapping column names or table purposes
        - Vague or overloaded terms in the query
        - Multiple possible paths through relationships
        - Implicit assumptions

        ### ✅ If Ambiguous:
        - Identify all valid interpretations
        - For each interpretation, include:
            - A short interpretation ID or title
            - A human-readable summary
            - The reasoning behind why it’s valid
            - Confidence level (high / medium / low)
            - A complete SQL query for that interpretation

        ### ❌ If Not Ambiguous:
        - Clearly explain why there is only one clear interpretation
        - Proceed to generate a single SQL query

        ---

        ## 🧠 Step-by-Step SQL Reasoning

        You must go through **each of the 15 provided tables** and their relationships before selecting which to use.

        ### Step 1: Identify the main table(s) that contain the core data required to answer the query  
        ### Step 2: Determine which related tables are required (JOINs) based on foreign key relationships  
        ### Step 3: Identify required filters/conditions based on the query and schema (including `unique_values`)  
        ### Step 4: Construct the SQL query logically and step-by-step, ensuring it is optimized and returns only relevant fields

        🛑 Guidelines:
        - Only retrieve columns directly relevant to answering the question
        - Avoid including UUIDs or technical/internal fields unless explicitly requested
        - Use values from `unique_values` for filtering whenever applicable

        ---

        ## 📤 Final Output Format (JSON)
        The output should be only the json structure, Return your result in the following JSON structure:
        ```json
        {{
            "ambiguous": true/false,
            "interpretations": [
                {{
                    "interpretation_id": "<short title or label>",
                    "user_friendly_title": "Human-friendly title of the interpretation",
                    "reasoning": "<explanation of why this interpretation is valid>",
                    "confidence": "<high / medium / low>",
                    "sql_query": "<final SQL query>"
                }}
            ]
        }}
        """

        prompt = stepwise_prompt.format(
            query=user_query,
            schema_info=json.dumps(schema_info, indent=2),
            fk_info=json.dumps(fk_info, indent=2)
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()

        print(response_text)

        json_match = re.search(r"json\s*({[\s\S]+?})\s*", response_text)

        if not json_match:
            raise ValueError("No valid JSON found in the response.")
        clean_response = json_match.group(1)


        return clean_response

    def get_final_sql_from_user_query(self, user_query: str, schema_info: dict, fk_info: dict):
        sql_response = self.decompose_query(
            user_query=user_query,
            schema_info=schema_info,
            fk_info=fk_info
        )

        parsed_response = json.loads(sql_response)

        if not parsed_response.get("ambiguous"):
            return parsed_response["interpretations"][0]["sql_query"]

        interpretations = parsed_response.get("interpretations", [])

        print("⚠️ Multiple possible interpretations found for your query:\n")
        for idx, interp in enumerate(interpretations, 1):
            print(f"{idx}. {interp['user_friendly_title']}")

        try:
            choice = int(input("Please choose the correct interpretation (enter number): ").strip())
            selected = interpretations[choice - 1]
        except (ValueError, IndexError):
            print("\n⚠️ Invalid choice. Trying to select the recommended interpretation...\n")
            selected = next((interp for interp in interpretations if interp.get("recommended")), interpretations[0])

        print(f"\n✅ You selected: {selected['user_friendly_title']}")
        return selected["sql_query"]