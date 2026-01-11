WITH earliest_admissions AS (
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
)

SELECT CASE 
           WHEN EXISTS (
               SELECT 1 
               FROM directed_edge_list 
               WHERE (subject_id1 = 10006580 AND subject_id2 = 10003400) 
                  OR (subject_id1 = 10003400 AND subject_id2 = 10006580)
           ) 
           THEN 1 ELSE 0 
       END AS path_exists;