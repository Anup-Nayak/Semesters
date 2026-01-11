SELECT a.subject_id, 
       a.hadm_id, 
       COUNT(DISTINCT p.icd_code) AS count_procedures, 
       COUNT(DISTINCT d.icd_code) AS count_diagnoses
FROM hosp.admissions a
LEFT JOIN hosp.procedures_icd p ON a.hadm_id = p.hadm_id
LEFT JOIN hosp.diagnoses_icd d ON a.hadm_id = d.hadm_id
WHERE a.admission_type = 'URGENT' 
AND a.hospital_expire_flag = 1
GROUP BY a.subject_id, a.hadm_id
ORDER BY a.subject_id, a.hadm_id, count_procedures DESC, count_diagnoses DESC;
