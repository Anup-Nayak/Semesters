WITH 
tmp1 AS (
    SELECT subject_id, hadm_id 
    FROM hosp.admissions 
    WHERE admission_type = 'URGENT' AND hospital_expire_flag = 1
),
tmp2 AS (
    SELECT subject_id, hadm_id, COUNT(*) AS count_diagnoses 
    FROM hosp.diagnoses_icd 
    GROUP BY subject_id, hadm_id
),
tmp3 AS (
    SELECT subject_id, hadm_id, COUNT(*) AS count_procedures 
    FROM hosp.procedures_icd 
    GROUP BY subject_id, hadm_id
)
SELECT distinct
    t1.subject_id, 
    t1.hadm_id, 
    COALESCE(t3.count_procedures, 0) AS count_procedures, 
    COALESCE(t2.count_diagnoses, 0) AS count_diagnoses
FROM 
    tmp1 AS t1
LEFT JOIN 
    tmp2 AS t2
ON 
    t1.subject_id = t2.subject_id AND t1.hadm_id = t2.hadm_id
LEFT JOIN 
    tmp3 AS t3
ON 
    t1.subject_id = t3.subject_id AND t1.hadm_id = t3.hadm_id
ORDER BY 
    t1.subject_id, 
    t1.hadm_id, 
    count_procedures DESC, 
    count_diagnoses DESC;
