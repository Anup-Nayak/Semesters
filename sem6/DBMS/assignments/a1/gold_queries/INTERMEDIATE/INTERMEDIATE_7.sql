with 
tmp1 as( select subject_id, min(admittime) as first_adm, max(admittime) as last_adm from hosp.admissions group by subject_id), 
diagnoses_match AS (
    SELECT 
        tmp1.subject_id
    FROM tmp1 
    JOIN hosp.diagnoses_icd di1
        ON tmp1.subject_id = di1.subject_id
    JOIN hosp.diagnoses_icd di2
        ON tmp1.subject_id = di2.subject_id
    WHERE di1.hadm_id = (SELECT hadm_id FROM hosp.admissions WHERE subject_id = tmp1.subject_id AND admittime = tmp1.first_adm)
      AND di2.hadm_id = (SELECT hadm_id FROM hosp.admissions WHERE subject_id = tmp1.subject_id AND admittime = tmp1.last_adm)
      AND di1.icd_code = di2.icd_code
      AND di1.icd_version = di2.icd_version
),
tmp6 as (select diagnoses_match.subject_id, hosp.patients.gender from diagnoses_match join hosp.patients on diagnoses_match.subject_id=hosp.patients.subject_id)
SELECT 
    gender,
    ROUND(COUNT(distinct subject_id) * 100.0 / SUM(COUNT(distinct subject_id)) OVER (), 2) AS percentage
FROM 
    tmp6
GROUP BY 
    gender
ORDER BY percentage DESC;