WITH first_admission AS (
SELECT 
    d.subject_id, 
    a.hadm_id AS first_hadm_id,
    TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') AS first_admit_time,
    TO_TIMESTAMP(a.dischtime, 'YYYY-MM-DD HH24:MI:SS') AS first_discharge_time
FROM hosp.diagnoses_icd d
JOIN hosp.admissions a 
    ON d.subject_id = a.subject_id AND d.hadm_id = a.hadm_id
WHERE d.icd_code LIKE 'I2%'
AND a.admittime = (
    SELECT MIN(admittime)
    FROM hosp.admissions a2
    WHERE a2.subject_id = a.subject_id
)
),second_admission AS (
    SELECT DISTINCT ON (fa.subject_id) 
        fa.subject_id,
        fa.first_hadm_id,
        a.hadm_id AS second_hadm_id,
        TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') AS second_admit_time,
        age(TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS'), fa.first_discharge_time) AS time_gap
    FROM first_admission fa
    JOIN hosp.admissions a 
        ON fa.subject_id = a.subject_id
        AND TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') > fa.first_discharge_time
        AND age(TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS'), fa.first_discharge_time) <= interval '180 days'
    ORDER BY fa.subject_id, TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') -- Earliest admission within 180 days
),

service_list AS (
SELECT 
    sa.subject_id,
    sa.second_hadm_id,
    ARRAY_REMOVE(ARRAY_AGG(t.curr_service ORDER BY t.transfertime), NULL) AS services
FROM second_admission sa    
LEFT JOIN hosp.services t 
    ON sa.subject_id = t.subject_id 
    AND sa.second_hadm_id = t.hadm_id
GROUP BY sa.subject_id, sa.second_hadm_id
)
SELECT 
sa.subject_id,
sa.second_hadm_id,
TO_CHAR(sa.time_gap, 'YYYY-MM-DD HH24:MI:SS') AS time_gap_between_admissions,
COALESCE(sl.services, '{}') AS services
FROM second_admission sa
LEFT JOIN service_list sl 
ON sa.subject_id = sl.subject_id AND sa.second_hadm_id = sl.second_hadm_id
ORDER BY 
CARDINALITY(sl.services) DESC,
sa.time_gap DESC,
sa.subject_id ASC,
sa.second_hadm_id ASC;
