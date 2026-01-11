SELECT a.hadm_id, 
       p.gender, 
       JUSTIFY_INTERVAL(TO_TIMESTAMP(a.dischtime, 'YYYY-MM-DD HH24:MI:SS') - 
                        TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS')) AS duration
FROM hosp.admissions a
JOIN hosp.patients p ON a.subject_id = p.subject_id
WHERE a.dischtime IS NOT NULL
ORDER BY duration, a.hadm_id;