WITH FirstLastDiagnosis AS (
    SELECT d.subject_id,
           d.hadm_id,
           icd.icd_code,
           icd.icd_version,
           icd.long_title AS diagnosis_title,
           a.admittime,
           RANK() OVER (PARTITION BY d.subject_id ORDER BY a.admittime ASC) AS first_rank,
           RANK() OVER (PARTITION BY d.subject_id ORDER BY a.admittime DESC) AS last_rank
    FROM hosp.diagnoses_icd d
    JOIN hosp.admissions a ON d.hadm_id = a.hadm_id
    JOIN hosp.d_icd_diagnoses icd 
        ON d.icd_code = icd.icd_code 
        AND d.icd_version = icd.icd_version
),
FirstDiagnosis AS (
    SELECT subject_id, 
           ARRAY_AGG(DISTINCT diagnosis_title ORDER BY diagnosis_title) AS diagnosis_set
    FROM FirstLastDiagnosis
    WHERE first_rank = 1
    GROUP BY subject_id
),
LastDiagnosis AS (
    SELECT subject_id, 
           ARRAY_AGG(DISTINCT diagnosis_title ORDER BY diagnosis_title) AS diagnosis_set
    FROM FirstLastDiagnosis
    WHERE last_rank = 1
    GROUP BY subject_id
),
MatchingPatients AS (
    SELECT p.subject_id, p.gender
    FROM FirstDiagnosis f
    JOIN LastDiagnosis l ON f.subject_id = l.subject_id
    JOIN hosp.patients p ON f.subject_id = p.subject_id
    WHERE EXISTS (
    SELECT 1 
    FROM unnest(f.diagnosis_set) AS first_diag
    JOIN unnest(l.diagnosis_set) AS last_diag 
    ON first_diag = last_diag
    )
),
SelectedCounts AS (
    SELECT gender, COUNT(*) AS selected_count
    FROM MatchingPatients
    GROUP BY gender
),
TotalSelected AS (
    SELECT SUM(selected_count) AS total_selected
    FROM SelectedCounts
)
SELECT sc.gender, 
       ROUND(100.0 * sc.selected_count / ts.total_selected, 2) AS percentage
FROM SelectedCounts sc
JOIN TotalSelected ts ON true
ORDER BY percentage DESC;