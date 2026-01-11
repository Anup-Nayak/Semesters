SELECT d.subject_id, 
       COUNT(DISTINCT a.hadm_id) AS count_admissions, 
       EXTRACT(YEAR FROM TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS')) AS year
FROM hosp.admissions a
JOIN hosp.diagnoses_icd d ON a.hadm_id = d.hadm_id
JOIN hosp.d_icd_diagnoses di ON d.icd_code = di.icd_code
WHERE LOWER(di.long_title) LIKE '%infection%'
GROUP BY d.subject_id, year
HAVING COUNT(DISTINCT a.hadm_id) > 1
ORDER BY year, count_admissions DESC, subject_id;