SELECT hadm_id, gender, 
       CASE 
           WHEN EXTRACT(DAY FROM (dischtime::timestamp - admittime::timestamp)) > 0 
           THEN EXTRACT(DAY FROM (dischtime::timestamp - admittime::timestamp)) || ' days, ' || TO_CHAR(dischtime::timestamp - admittime::timestamp, 'HH24:MI:SS')
           ELSE TO_CHAR(dischtime::timestamp - admittime::timestamp, 'HH24:MI:SS')
       END AS duration
FROM hosp.admissions 
JOIN hosp.patients ON hosp.patients.subject_id = hosp.admissions.subject_id 
ORDER BY (dischtime::TIMESTAMP - admittime::TIMESTAMP), hadm_id;