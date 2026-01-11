SELECT a.subject_id, 
       JUSTIFY_INTERVAL(AVG(TO_TIMESTAMP(a.dischtime, 'YYYY-MM-DD HH24:MI:SS') - 
                            TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS'))) 
       AS avg_duration
FROM hosp.admissions a
WHERE a.dischtime IS NOT NULL
GROUP BY a.subject_id
ORDER BY a.subject_id;