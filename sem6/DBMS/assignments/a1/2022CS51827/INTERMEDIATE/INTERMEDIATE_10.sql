WITH first_admissions AS (
    SELECT a.subject_id,  min(a.admittime) as admittime
    FROM hosp.admissions a
    JOIN hosp.diagnoses_icd d ON a.subject_id = d.subject_id
    JOIN hosp.d_icd_diagnoses di ON d.icd_code = di.icd_code
    AND d.icd_version = di.icd_version
    WHERE LOWER(di.long_title) LIKE '%kidney%'
    GROUP BY a.subject_id
    ORDER BY admittime desc
)
SELECT DISTINCT a.subject_id
FROM hosp.admissions a
JOIN first_admissions f ON a.subject_id = f.subject_id
WHERE a.admittime > f.admittime
ORDER BY a.subject_id
LIMIT 100;