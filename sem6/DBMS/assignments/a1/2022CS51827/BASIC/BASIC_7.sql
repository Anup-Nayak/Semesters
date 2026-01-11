SELECT a.subject_id, 
       max(a.hadm_id) AS latest_hadm_id, 
       max(p.dod) as dod
FROM hosp.admissions a
JOIN hosp.patients p ON a.subject_id = p.subject_id
WHERE dod IS NOT NULL
GROUP BY a.subject_id
ORDER BY a.subject_id;