WITH E10_E11_Patients AS (
    SELECT DISTINCT d.subject_id, d.hadm_id, a.admittime,
           LEAD(d.hadm_id) OVER (PARTITION BY d.subject_id ORDER BY a.admittime) AS next_hadm_id
    FROM hosp.diagnoses_icd d
    JOIN hosp.admissions a ON d.hadm_id = a.hadm_id
    WHERE d.icd_code LIKE 'E10%' OR d.icd_code LIKE 'E11%'
),
N18_Patients AS (
    SELECT DISTINCT d.subject_id, d.hadm_id
    FROM hosp.diagnoses_icd d
    WHERE d.icd_code LIKE 'N18%'
),
LinkedAdmissions AS (
    SELECT e.subject_id, e.hadm_id AS admission_id
    FROM E10_E11_Patients e
    JOIN N18_Patients n 
    ON e.subject_id = n.subject_id
    AND (e.hadm_id = n.hadm_id OR e.next_hadm_id = n.hadm_id)
),
Diagnoses AS (
    SELECT DISTINCT d.subject_id, d.hadm_id AS admission_id, d.icd_code, 'diagnoses' AS diagnoses_or_procedure
    FROM hosp.diagnoses_icd d
    JOIN LinkedAdmissions la ON d.subject_id = la.subject_id
),
Procedures AS (
    SELECT DISTINCT p.subject_id, p.hadm_id AS admission_id, p.icd_code, 'procedures' AS diagnoses_or_procedure
    FROM hosp.procedures_icd p
    JOIN LinkedAdmissions la ON p.subject_id = la.subject_id
)
SELECT subject_id, admission_id, diagnoses_or_procedure, icd_code
FROM (
    SELECT * FROM Diagnoses
    UNION
    SELECT * FROM Procedures
) AS combined_results
ORDER BY subject_id ASC, admission_id ASC, icd_code ASC, diagnoses_or_procedure ASC;