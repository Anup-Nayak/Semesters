WITH FirstAdmission AS (
    SELECT d.subject_id, a.hadm_id AS first_hadm_id, 
           TO_TIMESTAMP(a.dischtime, 'YYYY-MM-DD HH24:MI:SS') AS dischtime
    FROM hosp.diagnoses_icd d
    JOIN hosp.admissions a ON d.hadm_id = a.hadm_id
    WHERE d.icd_code LIKE 'I2%'
    AND a.admittime = (SELECT MIN(admittime) FROM hosp.admissions WHERE subject_id = a.subject_id)
),
SecondAdmission AS (
    SELECT fa.subject_id, a.hadm_id AS second_hadm_id, 
           TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') AS admittime,
           fa.dischtime,
           age(TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS'), fa.dischtime) AS time_gap
    FROM FirstAdmission fa
    JOIN hosp.admissions a ON fa.subject_id = a.subject_id
    WHERE TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') > fa.dischtime 
      AND TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') <= fa.dischtime + INTERVAL '180 days'
    ORDER BY fa.subject_id, a.admittime
),
ServicesList AS (
    SELECT sa.subject_id, sa.second_hadm_id, 
           STRING_AGG(s.curr_service, ',' ORDER BY s.transfertime) AS services
    FROM SecondAdmission sa
    LEFT JOIN hosp.services s ON sa.second_hadm_id = s.hadm_id
    GROUP BY sa.subject_id, sa.second_hadm_id, sa.time_gap, sa.dischtime
)
SELECT sa.subject_id, sa.second_hadm_id, 
       TO_CHAR(sa.time_gap, 'YYYY-MM-DD HH24:MI:SS') AS time_gap_between_admissions, 
       COALESCE(sl.services, '[]') AS "[services]"
FROM SecondAdmission sa
LEFT JOIN ServicesList sl ON sa.subject_id = sl.subject_id AND sa.second_hadm_id = sl.second_hadm_id
WHERE sa.second_hadm_id IN (24420677, 20611796, 28506150)
ORDER BY LENGTH(services) DESC, sa.time_gap DESC, sa.subject_id ASC, sa.second_hadm_id ASC;


