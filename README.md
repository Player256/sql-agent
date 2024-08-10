# SQL-Agent

## Overview

SQL-Agent is a multi-agent system designed to assist with SQL query generation and execution. It leverages Langgraph's multi-agent collaboration to interact with a database, generate accurate SQL queries, and provide results based on natural language questions.

## Features

- **Table Discovery**: Automatically fetches available tables from the database.
- **Relevance Determination**: Identifies which tables are relevant to the user's question.
- **DDL Fetching**: Retrieves the Data Definition Language (DDL) for the relevant tables.
- **Query Generation**: Generates SQL queries based on the user's question and the DDL of relevant tables.
- **Query Validation**: Uses a Language Model (LLM) to double-check the query for common mistakes.
- **Query Execution**: Executes the SQL query against the database and returns the results.
- **Error Handling**: Iteratively corrects any mistakes surfaced by the database engine until the query is successful.
- **Response Formulation**: Provides a clear response to the user based on the query results.

## End-to-end workflow
![image](https://github.com/user-attachments/assets/0a70299a-7aa6-4d28-8bda-8fae8a1e4a34)
