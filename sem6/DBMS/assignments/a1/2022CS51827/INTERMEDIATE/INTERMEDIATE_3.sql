SELECT c.caregiver_id, 
       COALESCE(p.procedureevents_count, 0) AS procedureevents_count,
       COALESCE(ch.chartevents_count, 0) AS chartevents_count,
       COALESCE(d.datetimeevents_count, 0) AS datetimeevents_count
FROM icu.caregiver AS c
LEFT JOIN (
    SELECT caregiver_id, COUNT(*) AS procedureevents_count
    FROM icu.procedureevents
    GROUP BY caregiver_id
) AS p ON p.caregiver_id = c.caregiver_id
LEFT JOIN (
    SELECT caregiver_id, COUNT(*) AS chartevents_count
    FROM icu.chartevents
    GROUP BY caregiver_id
) AS ch ON ch.caregiver_id = c.caregiver_id
LEFT JOIN (
    SELECT caregiver_id, COUNT(*) AS datetimeevents_count
    FROM icu.datetimeevents
    GROUP BY caregiver_id
) AS d ON d.caregiver_id = c.caregiver_id
ORDER BY c.caregiver_id, procedureevents_count, chartevents_count, datetimeevents_count;
