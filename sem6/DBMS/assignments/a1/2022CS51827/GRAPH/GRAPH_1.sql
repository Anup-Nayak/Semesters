WITH earliest_admissions AS (
    SELECT subject_id, hadm_id, admittime, dischtime
    FROM hosp.admissions
    ORDER BY TO_TIMESTAMP(admittime, 'YYYY-MM-DD HH24:MI:SS')
    LIMIT 200
),
overlapping_admissions AS (
    SELECT a1.subject_id AS subject_id1, 
           a2.subject_id AS subject_id2,
           a1.hadm_id AS hadm1,
           a2.hadm_id AS hadm2
    FROM earliest_admissions a1
    JOIN earliest_admissions a2 
    ON a1.subject_id < a2.subject_id
    AND TO_TIMESTAMP(a1.admittime, 'YYYY-MM-DD HH24:MI:SS') < TO_TIMESTAMP(a2.dischtime, 'YYYY-MM-DD HH24:MI:SS')
    AND TO_TIMESTAMP(a2.admittime, 'YYYY-MM-DD HH24:MI:SS') < TO_TIMESTAMP(a1.dischtime, 'YYYY-MM-DD HH24:MI:SS')
),
shared_diagnoses AS (
    SELECT DISTINCT d1.subject_id AS subject_id1, 
                    d2.subject_id AS subject_id2,
                    d1.hadm_id AS hadm1,
                    d2.hadm_id AS hadm2
    FROM hosp.diagnoses_icd d1
    JOIN hosp.diagnoses_icd d2 
    ON d1.subject_id < d2.subject_id  
    AND d1.icd_code = d2.icd_code
    AND d1.icd_version = d2.icd_version
    AND d1.hadm_id IN (
        SELECT hadm1 FROM overlapping_admissions oa
        WHERE oa.subject_id1 = d1.subject_id AND oa.subject_id2 = d2.subject_id
    )
    AND d2.hadm_id IN (
        SELECT hadm2 FROM overlapping_admissions oa
        WHERE oa.subject_id1 = d1.subject_id AND oa.subject_id2 = d2.subject_id
    )
)
SELECT DISTINCT o.subject_id1, o.subject_id2
FROM overlapping_admissions o
JOIN shared_diagnoses s ON o.subject_id1 = s.subject_id1 AND o.subject_id2 = s.subject_id2
ORDER BY o.subject_id1, o.subject_id2;