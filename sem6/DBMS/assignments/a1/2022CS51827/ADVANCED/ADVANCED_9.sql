WITH medication_patients AS (
    SELECT subject_id, hadm_id,
           CASE 
               WHEN COUNT(DISTINCT drug) = 2 THEN 'both'
               WHEN MAX(CASE WHEN LOWER(drug) = 'amlodipine' THEN 1 ELSE 0 END) = 1 THEN 'amlodipine'
               WHEN MAX(CASE WHEN LOWER(drug) = 'lisinopril' THEN 1 ELSE 0 END) = 1 THEN 'lisinopril'
           END AS drug
    FROM hosp.prescriptions
    WHERE LOWER(drug) IN ('amlodipine', 'lisinopril')
    GROUP BY subject_id, hadm_id
),
service_sequence AS (
    SELECT s.subject_id, s.hadm_id,
           ARRAY_TO_STRING(ARRAY_AGG(curr_service ORDER BY transfertime), ', ') AS services
    FROM hosp.services s
    GROUP BY s.subject_id, s.hadm_id
)
SELECT mp.subject_id, mp.hadm_id, mp.drug, COALESCE(ss.services, '') AS "[services]"
FROM medication_patients mp
LEFT JOIN service_sequence ss ON mp.subject_id = ss.subject_id AND mp.hadm_id = ss.hadm_id
ORDER BY mp.subject_id, mp.hadm_id;