WITH diabetes_patients AS (
    SELECT DISTINCT d.subject_id, d.hadm_id AS diabetes_hadm_id, 
           TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') AS diabetes_admit_time
    FROM hosp.diagnoses_icd d
    JOIN hosp.admissions a ON d.hadm_id = a.hadm_id
    WHERE d.icd_code LIKE 'E10%' OR d.icd_code LIKE 'E11%'
),

next_admissions AS (
    SELECT d.subject_id, d.diabetes_hadm_id, d.diabetes_admit_time, 
           a.hadm_id AS next_hadm_id, TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') AS next_admit_time,
           ROW_NUMBER() OVER (PARTITION BY d.subject_id, d.diabetes_hadm_id ORDER BY TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS')) AS rn
    FROM diabetes_patients d
    JOIN hosp.admissions a ON d.subject_id = a.subject_id
    WHERE TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') > d.diabetes_admit_time
),

immediate_next_admission AS (
    SELECT subject_id, diabetes_hadm_id, diabetes_admit_time, next_hadm_id, next_admit_time
    FROM next_admissions
    WHERE rn = 1
),

ckd_patients AS (
    SELECT DISTINCT d.subject_id, d.hadm_id AS ckd_hadm_id,
           TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') AS ckd_admit_time
    FROM hosp.diagnoses_icd d
    JOIN hosp.admissions a ON d.hadm_id = a.hadm_id
    WHERE d.icd_code LIKE 'N18%'
),

eligible_patients AS (
    SELECT d.subject_id, d.diabetes_hadm_id AS hadm_id
    FROM diabetes_patients d
    LEFT JOIN immediate_next_admission n ON (d.subject_id = n.subject_id AND d.diabetes_hadm_id = n.diabetes_hadm_id)   
    LEFT JOIN ckd_patients c ON d.subject_id = c.subject_id
    WHERE c.ckd_admit_time >= d.diabetes_admit_time AND c.ckd_admit_time <= COALESCE(n.next_admit_time, d.diabetes_admit_time)
),

diagnosis_codes AS (
    SELECT 
        e.subject_id,
        d.hadm_id,
        d.icd_code,
        'diagnoses' AS code_type
    FROM eligible_patients e
    JOIN hosp.diagnoses_icd d 
        ON e.subject_id = d.subject_id 
),

procedure_codes AS (
    SELECT 
        e.subject_id,
        p.hadm_id,
        p.icd_code,
        'procedures' AS code_type
    FROM eligible_patients e
    JOIN hosp.procedures_icd p 
        ON e.subject_id = p.subject_id 
)

SELECT DISTINCT subject_id, hadm_id, code_type, icd_code
FROM (
    SELECT * FROM diagnosis_codes
    UNION
    SELECT * FROM procedure_codes
) combined
ORDER BY subject_id, hadm_id, icd_code, code_type;
