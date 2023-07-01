-- Splitting the data into training set and test sets
-- Create temporary table to hold the balanced data
CREATE TEMPORARY TABLE balanced_data AS (
  SELECT *
  FROM your_table
  WHERE category IN (
    SELECT category
    FROM your_table
    GROUP BY category
    HAVING COUNT(*) > 1 -- Ensures at least two records per category
  )
);

-- Randomly assign each record to either the training or testing set
CREATE TEMPORARY TABLE training_set AS (
  SELECT *
  FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY category ORDER BY RANDOM()) AS row_num
    FROM balanced_data
  ) AS sub
  WHERE row_num <= CEIL(0.7 * (SELECT COUNT(*) FROM balanced_data))
);

CREATE TEMPORARY TABLE testing_set AS (
  SELECT *
  FROM balanced_data
  WHERE (category, id) NOT IN (
    SELECT category, id
    FROM training_set
  )
);

-- Output the training and testing sets
SELECT * FROM training_set;
SELECT * FROM testing_set;

-- test metrics
WITH class_counts AS (
    SELECT 
        actual_class,
        COUNT(*) AS total_count
    FROM 
        your_table
    GROUP BY 
        actual_class
),

true_positives AS (
    SELECT 
        actual_class,
        COUNT(*) AS tp_count
    FROM 
        your_table
    WHERE 
        predicted_class = actual_class
    GROUP BY 
        actual_class
),

predicted_positives AS (
    SELECT 
        predicted_class,
        COUNT(*) AS pp_count
    FROM 
        your_table
    GROUP BY 
        predicted_class
)

SELECT 
    (2 * SUM(tp_count) / CAST(SUM(tp_count + pp_count) AS FLOAT)) / CAST(SUM(total_count) AS FLOAT) AS f1_score,
    SUM(tp_count) / CAST(SUM(total_count) AS FLOAT) AS recall,
    SUM(tp_count) / CAST(SUM(pp_count) AS FLOAT) AS precision
FROM 
    class_counts
JOIN 
    true_positives USING (actual_class)
JOIN 
    predicted_positives USING (predicted_class);






