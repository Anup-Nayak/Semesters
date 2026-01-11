SELECT JUSTIFY_INTERVAL(AVG(TO_TIMESTAMP(a.dischtime, 'YYYY-MM-DD HH24:MI:SS') - 
                            TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS'))) 
       AS avg_duration
FROM hosp.admissions a
JOIN hosp.diagnoses_icd d ON a.hadm_id = d.hadm_id
WHERE d.icd_code = '4019' AND d.icd_version = 9
AND a.dischtime IS NOT NULL;