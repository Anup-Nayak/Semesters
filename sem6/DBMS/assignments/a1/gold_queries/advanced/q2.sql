WITH resistance_count AS (
    SELECT 
        hadm_id,
        subject_id,
        COUNT(DISTINCT micro_specimen_id) AS resistant_antibiotic_count
    FROM hosp.microbiologyevents
    WHERE interpretation = 'R' and 
    hadm_id IS NOT NULL
    GROUP BY hadm_id, subject_id
    HAVING COUNT(DISTINCT micro_specimen_id) >= 2
),

icu_info AS (
    SELECT 
        icu.subject_id, 
        icu.hadm_id,
        icu.stay_id,
        -- Convert intime and outtime from string to timestamp and calculate ICU length of stay in hours
        COALESCE(EXTRACT(EPOCH FROM (TO_TIMESTAMP(icu.outtime, 'YYYY-MM-DD HH24:MI:SS') - TO_TIMESTAMP(icu.intime, 'YYYY-MM-DD HH24:MI:SS'))) / 3600, 0) AS icu_los_hours  -- ICU length of stay in hours, default to 0 if no ICU stay
    FROM icu.icustays icu
),

mortality_info AS (
    SELECT 
        subject_id, 
        hadm_id, 
        CASE WHEN discharge_location = 'DIED' THEN 1 ELSE 0 END AS died_in_hospital
    FROM hosp.admissions
    WHERE hadm_id IS NOT NULL  -- Exclude admissions with null hadm_id
)

SELECT 
    r.subject_id, 
    r.hadm_id, 
    r.resistant_antibiotic_count, 
    ROUND(COALESCE(i.icu_los_hours, 0), 2) AS icu_length_of_stay_hours, 
    COALESCE(m.died_in_hospital, 0) AS died_in_hospital
FROM resistance_count r
LEFT JOIN icu_info i ON r.hadm_id = i.hadm_id
LEFT JOIN mortality_info m ON r.hadm_id = m.hadm_id
ORDER BY 
    died_in_hospital DESC, 
    resistant_antibiotic_count DESC, 
    icu_length_of_stay_hours DESC, 
    subject_id ASC, 
    hadm_id ASC;