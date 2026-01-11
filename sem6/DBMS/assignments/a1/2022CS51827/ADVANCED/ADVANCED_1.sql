WITH patient_drug_diagnosis AS (
    SELECT 
        a.subject_id,
        a.hadm_id,
        ARRAY_REMOVE(ARRAY_AGG(DISTINCT p.drug ORDER BY p.drug), NULL) AS medications,
        ARRAY_REMOVE(ARRAY_AGG(DISTINCT d.icd_code ORDER BY d.icd_code), NULL) AS diagnoses
    FROM 
        hosp.admissions a
    LEFT JOIN 
        hosp.prescriptions p ON a.hadm_id = p.hadm_id
    LEFT JOIN 
        hosp.diagnoses_icd d ON a.hadm_id = d.hadm_id
    GROUP BY 
        a.subject_id, a.hadm_id
)
SELECT
    subject_id,
    COUNT(DISTINCT hadm_id) AS total_admissions,
    COUNT(DISTINCT diagnoses) AS num_distinct_diagnoses_set_count,
    COUNT(DISTINCT medications) AS num_distinct_medications_set_count
FROM 
    patient_drug_diagnosis
GROUP BY 
    subject_id
HAVING 
    COUNT(DISTINCT diagnoses) >= 3 
    OR COUNT(DISTINCT medications) >= 3
ORDER BY 
    total_admissions DESC,
    num_distinct_diagnoses_set_count DESC,
    subject_id ASC;