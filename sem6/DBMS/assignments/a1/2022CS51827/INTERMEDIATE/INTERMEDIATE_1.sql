SELECT p.subject_id, 
       a.hadm_id AS hadm_id, 
       p.dod
FROM hosp.patients p
JOIN hosp.admissions a ON p.subject_id = a.subject_id
WHERE p.dod IS NOT NULL
AND TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') = (
    SELECT MIN(TO_TIMESTAMP(admittime, 'YYYY-MM-DD HH24:MI:SS'))
    FROM hosp.admissions
    WHERE subject_id = p.subject_id
)
ORDER BY p.subject_id;