-- MLflow keeps its own backend store separate from the application tables.
-- Runs before sql/init.sql (alphabetical order in docker-entrypoint-initdb.d).
CREATE DATABASE mlflow;

-- Airflow's metadata database, likewise kept separate.
CREATE DATABASE airflow;
