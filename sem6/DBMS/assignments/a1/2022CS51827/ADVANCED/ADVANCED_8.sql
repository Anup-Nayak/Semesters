WITH I10_Patients AS (
    SELECT DISTINCT d.subject_id, d.hadm_id, a.admittime,
           LEAD(d.hadm_id) OVER (PARTITION BY d.subject_id ORDER BY a.admittime) AS next_hadm_id
    FROM hosp.diagnoses_icd d
    JOIN hosp.admissions a ON d.hadm_id = a.hadm_id
    WHERE d.icd_code LIKE 'I10%'
),
I50_Patients AS (
    SELECT DISTINCT d.subject_id, d.hadm_id, a.admittime
    FROM hosp.diagnoses_icd d
    JOIN hosp.admissions a ON d.hadm_id = a.hadm_id
    WHERE d.icd_code LIKE 'I50%'
),
LinkedAdmissions AS (
    SELECT i10.subject_id, i10.hadm_id AS i10_hadm_id, i10.admittime AS i10_admittime,
           i50.hadm_id AS i50_hadm_id, i50.admittime AS i50_admittime
    FROM I10_Patients i10
    JOIN I50_Patients i50 
    ON i10.subject_id = i50.subject_id
    AND (i10.hadm_id = i50.hadm_id OR i10.next_hadm_id = i50.hadm_id)
),
FirstLastAdmissions AS (
    SELECT subject_id, 
           MIN(hadm_id) FILTER (WHERE icd_code LIKE 'I10%') AS first_hadm_id,  
           MAX(hadm_id) FILTER (WHERE icd_code LIKE 'I50%') AS last_hadm_id    
    FROM hosp.diagnoses_icd
    WHERE subject_id IN (SELECT subject_id FROM LinkedAdmissions)
    GROUP BY subject_id
    HAVING MIN(hadm_id) FILTER (WHERE icd_code LIKE 'I10%') IS NOT NULL
       AND MAX(hadm_id) FILTER (WHERE icd_code LIKE 'I50%') IS NOT NULL
),
IntermediateAdmissions AS (
    SELECT a.subject_id, a.hadm_id
    FROM hosp.admissions a
    JOIN FirstLastAdmissions fa ON a.subject_id = fa.subject_id
    WHERE a.admittime > (SELECT admittime FROM hosp.admissions WHERE hadm_id = fa.first_hadm_id)
      AND a.admittime < (SELECT admittime FROM hosp.admissions WHERE hadm_id = fa.last_hadm_id)
),
ValidPatients AS (
    SELECT ia.subject_id
    FROM IntermediateAdmissions ia
    GROUP BY ia.subject_id
    HAVING COUNT(DISTINCT ia.hadm_id) >= 2
),
DistinctDrugs AS (
    SELECT DISTINCT p.subject_id, p.hadm_id, p.drug
    FROM hosp.prescriptions p
    JOIN IntermediateAdmissions ia ON p.subject_id = ia.subject_id AND p.hadm_id = ia.hadm_id
    JOIN ValidPatients vp ON ia.subject_id = vp.subject_id
)
SELECT subject_id, hadm_id AS admission_id, drug
FROM DistinctDrugs
ORDER BY subject_id ASC, admission_id ASC, drug ASC;