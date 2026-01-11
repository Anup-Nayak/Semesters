WITH Procedures AS (
    SELECT DISTINCT p.subject_id, p.hadm_id, p.icd_code, p.icd_version, a.admittime, p.chartdate
    FROM hosp.procedures_icd p
    JOIN hosp.admissions a ON p.hadm_id = a.hadm_id
    WHERE (p.icd_code LIKE '0%' OR p.icd_code LIKE '1%' OR p.icd_code LIKE '2%')
      AND p.subject_id IS NOT NULL AND p.hadm_id IS NOT NULL
),
Medications AS (
    SELECT DISTINCT pr.subject_id, pr.hadm_id, pr.drug, pr.starttime::timestamp
    FROM hosp.prescriptions pr
    WHERE pr.hadm_id IS NOT NULL AND pr.subject_id IS NOT NULL
),
ValidAdmissions AS (
    SELECT DISTINCT proc.subject_id, proc.hadm_id, proc.chartdate
    FROM Procedures proc
    JOIN Medications med ON proc.hadm_id = med.hadm_id
    WHERE med.starttime::date BETWEEN proc.chartdate::date AND proc.chartdate::date + INTERVAL '1 day'
      AND proc.hadm_id IS NOT NULL AND proc.subject_id IS NOT NULL
),
DistinctMedications AS (
    SELECT subject_id, hadm_id, COUNT(DISTINCT drug) AS med_count
    FROM Medications
    WHERE hadm_id IS NOT NULL AND subject_id IS NOT NULL
    GROUP BY subject_id, hadm_id
    HAVING COUNT(DISTINCT drug) >= 2
),
FinalAdmissions AS (
    SELECT v.subject_id, v.hadm_id, v.chartdate
    FROM ValidAdmissions v
    JOIN DistinctMedications dm ON v.subject_id = dm.subject_id AND v.hadm_id = dm.hadm_id
),
TimeGap AS (
    SELECT f.subject_id, f.hadm_id, 
           TO_CHAR(
               age(MAX(med.starttime), MIN(f.chartdate::timestamp)), 
               'YYYY-MM-DD HH24:MI:SS'
           ) AS time_gap
    FROM FinalAdmissions f
    JOIN Medications med ON f.subject_id = med.subject_id AND f.hadm_id = med.hadm_id
    GROUP BY f.subject_id, f.hadm_id
),
DistinctCounts AS (
    SELECT f.subject_id, f.hadm_id, 
           COUNT(DISTINCT diag.icd_code) AS distinct_diagnoses,
           COUNT(DISTINCT proc.icd_code) AS distinct_procedures
    FROM FinalAdmissions f
    JOIN hosp.diagnoses_icd diag ON f.subject_id = diag.subject_id AND f.hadm_id = diag.hadm_id
    JOIN hosp.procedures_icd proc ON f.subject_id = proc.subject_id AND f.hadm_id = proc.hadm_id
    GROUP BY f.subject_id, f.hadm_id
)
SELECT DISTINCT d.subject_id, d.hadm_id, d.distinct_diagnoses, d.distinct_procedures, t.time_gap
FROM DistinctCounts d
JOIN TimeGap t ON d.subject_id = t.subject_id AND d.hadm_id = t.hadm_id
ORDER BY d.distinct_diagnoses DESC, d.distinct_procedures DESC, t.time_gap ASC, d.subject_id ASC, d.hadm_id ASC;