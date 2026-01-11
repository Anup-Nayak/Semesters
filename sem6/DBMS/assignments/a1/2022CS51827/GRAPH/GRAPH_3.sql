WITH RECURSIVE earliest_admissions AS (
    SELECT subject_id, hadm_id, admittime, dischtime
    FROM hosp.admissions
    ORDER BY TO_TIMESTAMP(admittime, 'YYYY-MM-DD HH24:MI:SS')
    LIMIT 200
),
directed_edges AS (
    SELECT DISTINCT a1.subject_id AS subject_id1, 
                    a2.subject_id AS subject_id2
    FROM earliest_admissions a1
    JOIN earliest_admissions a2 
    ON a1.subject_id <> a2.subject_id
    WHERE 
    TO_TIMESTAMP(a1.admittime, 'YYYY-MM-DD HH24:MI:SS') 
        < TO_TIMESTAMP(a2.dischtime, 'YYYY-MM-DD HH24:MI:SS')
    AND 
    TO_TIMESTAMP(a1.dischtime, 'YYYY-MM-DD HH24:MI:SS') 
        > TO_TIMESTAMP(a2.admittime, 'YYYY-MM-DD HH24:MI:SS')
),
directed_edge_list AS (
    SELECT subject_id1, subject_id2
    FROM directed_edges
    ORDER BY subject_id1, subject_id2
),
shortest_path AS (
    
    SELECT subject_id1 AS start_id, subject_id2 AS end_id, 1 AS path_length
    FROM directed_edge_list
    WHERE subject_id1 = 10038081
    
    UNION ALL

    
    SELECT sp.start_id, de.subject_id2, sp.path_length + 1
    FROM shortest_path sp
    JOIN directed_edge_list de ON sp.end_id = de.subject_id1
    WHERE sp.path_length < 20
)
SELECT MIN(path_length) AS path_length
FROM shortest_path
WHERE end_id = 10021487;