
WITH yearly_admissions AS (
    SELECT 
        hosp.admissions.subject_id, 
        hosp.admissions.hadm_id, 
        hosp.admissions.admittime, 
        hosp.diagnoses_icd.icd_code, 
        hosp.diagnoses_icd.icd_version, 
        hosp.d_icd_diagnoses.long_title
    FROM 
        hosp.admissions 
    JOIN 
        hosp.diagnoses_icd 
        ON hosp.admissions.subject_id = hosp.diagnoses_icd.subject_id 
        AND hosp.admissions.hadm_id = hosp.diagnoses_icd.hadm_id 
    JOIN 
        hosp.d_icd_diagnoses 
        ON hosp.diagnoses_icd.icd_code = hosp.d_icd_diagnoses.icd_code
    WHERE 
        LOWER(hosp.d_icd_diagnoses.long_title) LIKE '%infection%' 
),
patient_admission_counts AS (
    SELECT 
        subject_id, 
        COUNT(DISTINCT hadm_id) AS count_admissions, 
        DATE_PART('year', TO_TIMESTAMP(admittime, 'YYYY-MM-DD HH24:MI:SS'))::INTEGER as year
    FROM 
        yearly_admissions
    GROUP BY 
        subject_id, year
    HAVING 
        COUNT(DISTINCT hadm_id) > 1 
)
SELECT *
FROM 
    patient_admission_counts
ORDER BY 
    year, count_admissions DESC, subject_id;