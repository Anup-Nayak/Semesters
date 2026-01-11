WITH resistant_patients AS (
    SELECT subject_id, 
    hadm_id, 
    COUNT(DISTINCT micro_specimen_id) AS resistant_antibiotic_count
    
    FROM hosp.microbiologyevents
    
    WHERE interpretation = 'R' AND hadm_id IS NOT NULL
    GROUP BY subject_id, hadm_id
    HAVING COUNT(DISTINCT micro_specimen_id) >= 2
),
icu_stay AS (
    SELECT subject_id,
        hadm_id, 
            CASE
                WHEN icu.hadm_id is null THEN 0
                ELSE round(
                    extract(
                        epoch
                        from (icu.outtime::timestamp - icu.intime::timestamp)
                    ) / 3600.0,
                    2
                )
        END as icu_length_of_stay_hours
    FROM icu.icustays as icu
    GROUP BY subject_id, hadm_id, icu.outtime, icu.intime
)
SELECT r.subject_id, 
    r.hadm_id, 
    r.resistant_antibiotic_count, 
    COALESCE(i.icu_length_of_stay_hours, 0) AS icu_length_of_stay_hours, 
    CASE WHEN a.discharge_location = 'DIED' THEN 1 ELSE 0 END AS died_in_hospital
FROM resistant_patients r
LEFT JOIN icu_stay i ON r.subject_id = i.subject_id AND r.hadm_id = i.hadm_id
JOIN hosp.admissions a ON r.hadm_id = a.hadm_id
ORDER BY died_in_hospital DESC, 
        resistant_antibiotic_count DESC, 
        icu_length_of_stay_hours DESC, 
        r.subject_id, 
        r.hadm_id;