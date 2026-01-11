WITH i10_patients AS (
    SELECT DISTINCT d.subject_id, d.hadm_id AS i10_hadm_id, 
           TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') AS i10_admit_time
    FROM hosp.diagnoses_icd d
    JOIN hosp.admissions a ON d.hadm_id = a.hadm_id
    WHERE d.icd_code LIKE 'I10%' 
),

next_admissions AS (
    SELECT d.subject_id, d.i10_hadm_id, d.i10_admit_time, 
           a.hadm_id AS next_hadm_id, TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') AS next_admit_time,
           ROW_NUMBER() OVER (PARTITION BY d.subject_id, d.i10_hadm_id ORDER BY TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS')) AS rn
    FROM i10_patients d
    JOIN hosp.admissions a ON d.subject_id = a.subject_id
    WHERE TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') > d.i10_admit_time
),

immediate_next_admission AS (
    SELECT subject_id, i10_hadm_id, i10_admit_time, next_hadm_id, next_admit_time
    FROM next_admissions
    WHERE rn = 1
),

i50_patients AS (
    SELECT DISTINCT d.subject_id, d.hadm_id AS i50_hadm_id,
           TO_TIMESTAMP(a.admittime, 'YYYY-MM-DD HH24:MI:SS') AS i50_admit_time
    FROM hosp.diagnoses_icd d
    JOIN hosp.admissions a ON d.hadm_id = a.hadm_id
    WHERE d.icd_code LIKE 'I50%'
),

eligible_patients AS (
    SELECT DISTINCT d.subject_id
    FROM i10_patients d
    LEFT JOIN immediate_next_admission n ON (d.subject_id = n.subject_id AND d.i10_hadm_id = n.i10_hadm_id)   
    LEFT JOIN i50_patients c ON d.subject_id = c.subject_id
    WHERE c.i50_admit_time >= d.i10_admit_time AND c.i50_admit_time <= COALESCE(n.next_admit_time, d.i10_admit_time)
),

i10_diag AS (
    SELECT d.subject_id, d.hadm_id, TO_DATE(a.admittime, 'YYYY-MM-DD') AS chartdate
    FROM eligible_patients ep
    JOIN hosp.admissions a ON ep.subject_id = a.subject_id
    JOIN hosp.diagnoses_icd d ON a.hadm_id = d.hadm_id
    WHERE d.icd_code LIKE 'I10%'
),

i50_diag AS (
    SELECT d.subject_id, d.hadm_id, TO_DATE(a.admittime, 'YYYY-MM-DD') AS chartdate
    FROM eligible_patients ep
    JOIN hosp.admissions a ON ep.subject_id = a.subject_id
    JOIN hosp.diagnoses_icd d ON a.hadm_id = d.hadm_id
    WHERE d.icd_code LIKE 'I50%'
),

drug_list AS (
    SELECT DISTINCT p.subject_id, p.hadm_id, m.drug
    FROM hosp.prescriptions m
    JOIN hosp.admissions p ON m.hadm_id = p.hadm_id
    JOIN i10_diag i10 ON p.subject_id = i10.subject_id
    JOIN i50_diag i50 ON p.subject_id = i50.subject_id
    WHERE TO_DATE(p.admittime, 'YYYY-MM-DD') > i10.chartdate
      AND TO_DATE(p.admittime, 'YYYY-MM-DD') < i50.chartdate
)

SELECT subject_id, hadm_id, drug
FROM drug_list
ORDER BY subject_id ASC, hadm_id ASC, drug ASC;
