WITH procedure_info AS (
    SELECT 
        p.subject_id, 
        p.hadm_id, 
        TO_DATE(p.chartdate, 'YYYY-MM-DD') AS procedure_date,
        p.icd_code AS procedure_icd
    FROM hosp.procedures_icd p
    WHERE p.icd_code LIKE '0%' OR p.icd_code LIKE '1%' OR p.icd_code LIKE '2%'
),

medication_info AS (
    SELECT 
        m.subject_id, 
        m.hadm_id, 
        CASE 
            WHEN m.starttime ~ '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$' 
            THEN TO_TIMESTAMP(m.starttime, 'YYYY-MM-DD HH24:MI:SS')
            ELSE NULL
        END AS medication_starttime
    FROM hosp.prescriptions m
),

same_or_next_day AS (
    SELECT DISTINCT pi.subject_id, pi.hadm_id
    FROM procedure_info pi
    JOIN medication_info mi 
        ON pi.subject_id = mi.subject_id 
        AND pi.hadm_id = mi.hadm_id
    WHERE DATE(mi.medication_starttime) = pi.procedure_date 
          OR DATE(mi.medication_starttime) = pi.procedure_date + INTERVAL '1 day'
),

more_than_one_med AS (
    SELECT subject_id, hadm_id, COUNT(DISTINCT drug) AS distinct_medications_count
    FROM hosp.prescriptions
    GROUP BY subject_id, hadm_id
    HAVING COUNT(DISTINCT drug) > 1
),

first_proc AS (
    SELECT subject_id, hadm_id, icd_code, 
           TO_DATE(chartdate, 'YYYY-MM-DD') AS chartdate
    FROM (
        SELECT subject_id, hadm_id, icd_code, chartdate,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY chartdate ASC) AS rn
        FROM hosp.procedures_icd
    ) sub
    WHERE rn = 1
),

last_med AS (
    SELECT subject_id, hadm_id, drug, 
           TO_TIMESTAMP(starttime, 'YYYY-MM-DD HH24:MI:SS') AS starttime
    FROM (
        SELECT subject_id, hadm_id, drug, starttime,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY starttime DESC) AS rn
        FROM hosp.prescriptions
    ) sub
    WHERE rn = 1
),  

timegap AS (
    SELECT fp.subject_id, fp.hadm_id, 
           AGE(lm.starttime, TO_TIMESTAMP(fp.chartdate || ' 00:00:00', 'YYYY-MM-DD HH24:MI:SS')) AS time_gap
    FROM first_proc fp
    JOIN last_med lm ON fp.subject_id = lm.subject_id AND fp.hadm_id = lm.hadm_id
),

distinct_diagnoses AS (
    SELECT subject_id, hadm_id, COUNT(DISTINCT icd_code) AS distinct_diagnoses
    FROM hosp.diagnoses_icd
    GROUP BY subject_id, hadm_id
),

distinct_procedures AS (
    SELECT subject_id, hadm_id, COUNT(DISTINCT icd_code) AS distinct_procedures
    FROM hosp.procedures_icd
    GROUP BY subject_id, hadm_id
)

SELECT snd.subject_id, snd.hadm_id, 
       dd.distinct_diagnoses, 
       dp.distinct_procedures, 
       TO_CHAR(tg.time_gap, 'YYYY-MM-DD HH24:MI:SS') AS time_gap
FROM same_or_next_day snd
JOIN more_than_one_med mm ON snd.subject_id = mm.subject_id AND snd.hadm_id = mm.hadm_id
JOIN timegap tg ON snd.subject_id = tg.subject_id AND snd.hadm_id = tg.hadm_id
JOIN distinct_diagnoses dd ON snd.subject_id = dd.subject_id AND snd.hadm_id = dd.hadm_id
JOIN distinct_procedures dp ON snd.subject_id = dp.subject_id AND snd.hadm_id = dp.hadm_id
ORDER BY  dd.distinct_diagnoses DESC, dp.distinct_procedures DESC, tg.time_gap ASC, snd.subject_id ASC, snd.hadm_id ASC;
