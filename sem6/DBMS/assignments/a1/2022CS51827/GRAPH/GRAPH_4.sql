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
reachable_nodes AS (
    SELECT subject_id2 AS connected_id 
    FROM directed_edge_list 
    WHERE subject_id1 = 10038081
    
    UNION

    SELECT de.subject_id2 
    FROM reachable_nodes rn
    JOIN directed_edge_list de ON rn.connected_id = de.subject_id1
)
SELECT COUNT(DISTINCT connected_id) AS count FROM reachable_nodes;