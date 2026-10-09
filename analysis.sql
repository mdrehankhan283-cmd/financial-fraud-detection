-- SQLite: import artifacts/test_predictions.csv as test_predictions first.
SELECT Class, COUNT(*) AS transactions, AVG(Amount) AS average_amount
FROM test_predictions GROUP BY Class;
SELECT alert, Class, COUNT(*) AS transactions
FROM test_predictions GROUP BY alert, Class;
SELECT CAST(Time / 3600 AS INTEGER) AS hour_since_start,
       COUNT(*) AS transactions, SUM(Class) AS frauds, SUM(alert) AS alerts
FROM test_predictions GROUP BY hour_since_start ORDER BY hour_since_start;
