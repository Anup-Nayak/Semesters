WITH prescribed_patients AS (
    SELECT p.subject_id, p.hadm_id,
        CASE 
            WHEN COUNT(DISTINCT CASE WHEN LOWER(p.drug) = 'amlodipine' THEN p.drug END) > 0
             AND COUNT(DISTINCT CASE WHEN LOWER(p.drug) = 'lisinopril' THEN p.drug END) > 0 
            THEN 'both'
            WHEN COUNT(DISTINCT CASE WHEN LOWER(p.drug) = 'amlodipine' THEN p.drug END) > 0 
            THEN 'amlodipine'
            WHEN COUNT(DISTINCT CASE WHEN LOWER(p.drug) = 'lisinopril' THEN p.drug END) > 0 
            THEN 'lisinopril'
        END AS drug
    FROM hosp.prescriptions p
    WHERE LOWER(p.drug) IN ('amlodipine', 'lisinopril')
    GROUP BY p.subject_id, p.hadm_id
),
service_path AS (
    SELECT t.subject_id, t.hadm_id, 
           ARRAY_REMOVE(ARRAY_AGG(t.curr_service ORDER BY t.transfertime), NULL) AS services
    FROM hosp.services t
    WHERE t.curr_service IS NOT NULL
    GROUP BY t.subject_id, t.hadm_id
)
SELECT pp.subject_id, pp.hadm_id, pp.drug, 
       COALESCE(sp.services, '{}') AS services
FROM prescribed_patients pp
LEFT JOIN service_path sp ON pp.subject_id = sp.subject_id AND pp.hadm_id = sp.hadm_id
ORDER BY pp.subject_id, pp.hadm_id;
